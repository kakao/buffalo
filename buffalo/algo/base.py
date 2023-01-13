import abc
import logging
import pickle
import struct

import absl.logging
import numpy as np

logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

from buffalo.misc import aux

EPS = 1e-8


class Algo(abc.ABC):
    def __init__(self, *args, **kwargs):
        self._idmanager = aux.Option({"userid": [], "userid_map": {},
                                      "itemid": [], "itemid_map": {},
                                      "userid_mapped": False, "itemid_mapped": False})

    def get_option(self, opt_path) -> (aux.Option, str):
        if isinstance(opt_path, (dict, aux.Option)):
            opt_path = self.create_temporary_option_from_dict(opt_path)
        opt = aux.Option(opt_path)
        opt_path = opt_path
        self.is_valid_option(opt)
        return (aux.Option(opt), opt_path)

    def _normalize(self, feat):
        feat = feat / np.sqrt((feat ** 2).sum(-1) + EPS)[..., np.newaxis]
        return feat

    def initialize(self):
        self.__early_stopping = {"round": 0,
                                 "min_loss": 987654321}
        if self.opt.random_seed:
            np.random.seed(self.opt.random_seed)

    @abc.abstractmethod
    def normalize(self, group="item"):
        raise NotImplementedError

    def _get_topk_recommendation(self, p, Q, pb, Qb, pool, topk, num_workers):
        if pool is not None:
            Q = Q[pool]
            if Qb is not None:
                Qb = Qb[pool]

        scores = p.dot(Q.T)
        if pb is not None:
            scores += pb
        if Qb is not None:
            scores += Qb.T

        topks = self.get_topk(scores, k=topk, num_threads=num_workers)
        if pool is not None:
            topks = np.array([pool[t] for t in topks])
        return topks

    def topk_recommendation(self, keys, topk=10, pool=None):
        """Return TopK recommendation for each users(keys)

        :param keys: Query key(s)
        :type keys: list or str
        :param int topk: Number of recommendation
        :param pool: See the pool parameter of `most_similar`
        :rtype: dict or list
        """
        is_many = isinstance(keys, list)
        if not is_many:
            keys = [keys]
        if not self._idmanager.userid_mapped:
            self.build_userid_map()
        if not self._idmanager.itemid_mapped:
            self.build_itemid_map()
        if pool is not None:
            pool = self.get_index_pool(pool, group="item")
            if len(pool) == 0:
                return []
        rows = [self._idmanager.userid_map[k] for k in keys
                if k in self._idmanager.userid_map]
        topks = self._get_topk_recommendation(rows, topk, pool)
        if not topks:
            return []
        if is_many:
            return {self._idmanager.userids[k]: [self._idmanager.itemids[v] for v in vv]
                    for k, vv in topks}
        else:
            for k, vv in topks:
                return [self._idmanager.itemids[v] for v in vv]

    def most_similar(self, key, topk=10, group="item", pool=None):
        """Return top-k most similar items

        :param str key: Query key
        :param int topk: The number of results (default: 10)
        :param str group: Data group where to find (default: item)
        :param pool: The list of item keys to find for.
            If it is a numpy.ndarray instance then it treat as index of items and it would be helpful for calculation speed. (default: None)
        :type pool: list or numpy.ndarray
        :return: Top-k most similar items for given query.
        :rtype: list
        """
        if group == "item":
            if not self._idmanager.itemid_mapped:
                self.build_itemid_map()
            return self._most_similar_item(key, topk, pool)
        return []

    def _get_most_similar_item(self, col, topk, Factor, nrz, pool):
        if isinstance(col, np.ndarray):
            q = col
        else:
            topk += 1
            q = Factor[col]
        if nrz:
            if pool is not None:
                dot = q.dot(Factor[pool].T)
            else:
                dot = q.dot(Factor.T)
            # topks = np.argsort(dot)[-topk:][::-1]
            topks = self.get_topk(dot, k=topk, num_threads=self.opt.num_workers)
        else:
            if pool is not None:
                dot = q.dot(Factor[pool].T)
                dot = dot / (np.linalg.norm(q) * np.linalg.norm(Factor[pool], axis=1) + EPS)
            else:
                dot = q.dot(Factor.T)
                dot = dot / (np.linalg.norm(q) * np.linalg.norm(Factor, axis=1) + EPS)
            # topks = np.argsort(dot)[-topk:][::-1]
            topks = self.get_topk(dot, k=topk, num_threads=self.opt.num_workers)
        scores = dot[topks]
        if pool is not None:
            topks = np.array([pool[t] for t in topks])
        return topks, scores

    def _most_similar_item(self, key, topk=10, pool=None):
        is_vector = False
        if isinstance(key, np.ndarray):
            f = key
            is_vector = True
        else:
            col = self._idmanager.itemid_map.get(key)
            if col is None:
                return []
            f = col
        if pool is not None:
            pool = self.get_index_pool(pool, group="item")
            if len(pool) == 0:
                return []
        topks, scores = self._get_most_similar_item(f, topk, pool)
        if is_vector:
            return [(self._idmanager.itemids[k], v)
                    for (k, v) in zip(topks, scores)]
        else:
            return [(self._idmanager.itemids[k], v)
                    for (k, v) in zip(topks, scores)
                    if k != f]

    def build_itemid_map(self):
        idmap = self.data.get_group("idmap")
        header = self.data.get_header()
        if idmap["cols"].shape[0] == 0:
            self._idmanager.itemids = list(map(str, list(range(header["num_items"]))))
            self._idmanager.itemid_map = {str(i): i for i in range(header["num_items"])}
        else:
            self._idmanager.itemids = list(map(lambda x: x.decode("utf-8", "ignore"), idmap["cols"][::]))
            self._idmanager.itemid_map = {v: idx
                                          for idx, v in enumerate(self._idmanager.itemids)}
        self._idmanager.itemid_mapped = True

    def build_userid_map(self):
        idmap = self.data.get_group("idmap")
        header = self.data.get_header()
        if idmap["rows"].shape[0] == 0:
            self._idmanager.userids = list(map(str, list(range(header["num_users"]))))
            self._idmanager.userid_map = {str(i): i for i in range(header["num_users"])}
        else:
            self._idmanager.userids = list(map(lambda x: x.decode("utf-8", "ignore"), idmap["rows"][::]))
            self._idmanager.userid_map = {v: idx
                                          for idx, v in enumerate(self._idmanager.userids)}
        self._idmanager.userid_mapped = True

    def get_feature(self, name, group="item"):
        index = self.get_index(name, group=group)
        if index is None:
            return None
        return self._get_feature(index, group)

    @abc.abstractmethod
    def _get_feature(self, index, group="item"):
        raise NotImplementedError

    def get_weighted_feature(self, weights, group="item", min_length=1):
        if isinstance(weights, dict):
            feat = [(self.get_feature(k), w) for k, w in weights.items()]
            feat = [f * w for f, w in feat if f is not None]
        elif isinstance(weights, list):
            feat = [self.get_feature(k) for k, w in weights]
        if len(feat) < min_length:
            return None
        feat = np.array(feat, dtype=np.float64).mean(axis=0)
        return (feat / np.linalg.norm(feat) + EPS).astype(np.float32)

    def periodical(self, period, current):
        if not period or (current + 1) % period == 0:
            return True
        return False

    def save_best_only(self, loss, best_loss, i):
        if self.opt.save_best and best_loss > loss and self.periodical(self.opt.save_period, i):
            self.save(self.opt.model_path)
            return loss
        return best_loss

    def early_stopping(self, loss):
        if self.opt.early_stopping_rounds < 1:
            return False
        if self.__early_stopping["min_loss"] < loss:
            self.__early_stopping["round"] += 1
        else:
            self.__early_stopping["round"] = 0
        self.__early_stopping["min_loss"] = loss
        if self.__early_stopping["round"] >= self.opt.early_stopping_rounds:
            self.logger.info("Reached at early_stopping rounds, stopping train.")
            return True
        return False

    def get_index(self, keys, group="item"):
        """Get index list of given item keys.
        If there is no index for such key, return None.

        :param keys: Query key(s)
        :type keys: str or list
        :param str group: Data group where to find (default: item)
        :rtype: int or list
        """
        is_many = isinstance(keys, list)
        if not is_many:
            keys = [keys]
        indexes = []
        if group == "item":
            if not self._idmanager.itemid_mapped:
                self.build_itemid_map()
            indexes = [self._idmanager.itemid_map.get(k) for k in keys]
        elif group == "user":
            if not self._idmanager.userid_mapped:
                self.build_userid_map()
            indexes = [self._idmanager.userid_map.get(k) for k in keys]
        if not is_many:
            return indexes[0]
        return np.array(indexes)

    # NOTE: Ugly naming?
    def get_index_pool(self, pool, group="item"):
        """Simple wrapper of get_index.
        For np.ndarray pool, it returns asis with nothing. But list, it perform get_index with keys in pool.

        :param pool: The list of keys.
        :param str group: Data group where to find (default: item)
        :rtype: np.ndarray
        """
        if isinstance(pool, list):
            pool = self.get_index(pool, group)
            pool = np.array([p for p in pool if p is not None])
        elif isinstance(pool, np.ndarray):
            pass
        else:
            raise ValueError("Unexpected type for pool: %s" % type(pool))
        assert isinstance(pool, np.ndarray)
        return pool


class Serializable(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    def save(self, path=None, with_itemid_map=True, with_userid_map=True, data_fields=[]):
        if path is None:
            path = self.opt.model_path
        if with_itemid_map and not self._idmanager.itemid_mapped:
            self.build_itemid_map()
        if with_userid_map and not self._idmanager.userid_mapped:
            self.build_userid_map()
        data = self._get_data()
        if data_fields:
            data = [(k, v) for k, v in data if k in data_fields]
        with open(path, "wb") as fout:
            total_objs = len(data)
            fout.write(struct.pack("Q", total_objs))
            for name, obj in data:
                bname = bytes(name, encoding="utf-8")
                fout.write(struct.pack("Q", len(bname)))
                fout.write(bname)
                s = pickle.dumps(obj, protocol=4)
                fout.write(struct.pack("Q", len(s)))
                fout.write(s)

    def _get_data(self):
        data = [("_idmanager", self._idmanager)]
        return data

    def load(self, path, data_fields=[]):
        with open(path, "rb") as fin:
            total_objs = struct.unpack("Q", fin.read(8))[0]
            for _ in range(total_objs):
                name_sz = struct.unpack("Q", fin.read(8))[0]
                name = fin.read(name_sz).decode("utf8")
                obj_sz = struct.unpack("Q", fin.read(8))[0]
                if data_fields and name not in data_fields:
                    fin.seek(obj_sz, 1)
                    continue
                obj = pickle.loads(fin.read(obj_sz))
                setattr(self, name, obj)

    @classmethod
    def instantiate(cls, cls_opt, path, data_fields):
        opt = cls_opt().get_default_option()
        c = cls(opt)
        c.load(path, data_fields)
        return c
