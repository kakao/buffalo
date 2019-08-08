# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import abc
import json
import pickle
import struct
import logging
import datetime
from contextlib import suppress

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
# what the...
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

from buffalo.misc import aux


class Algo(abc.ABC):
    def __init__(self, *args, **kwargs):
        self.temporary_files = []
        self._idmanager = aux.Option({'userid': [], 'userid_map': {},
                                      'itemid': [], 'itemid_map': {},
                                      'userid_mapped': False, 'itemid_mapped': False})

    def __del__(self):
        with suppress(Exception):
            for path in self.temporary_files:
                os.remove(path)

    def get_option(self, opt_path) -> (aux.Option, str):
        if isinstance(opt_path, (dict, aux.Option)):
            opt_path = self.create_temporary_option_from_dict(opt_path)
            self.temporary_files.append(opt_path)
        opt = aux.Option(opt_path)
        opt_path = opt_path
        self.is_valid_option(opt)
        return (aux.Option(opt), opt_path)

    def _normalize(self, feat):
        feat = feat / np.sqrt((feat ** 2).sum(-1) + 1e-8)[..., np.newaxis]
        return feat

    @abc.abstractmethod
    def normalize(self, group='item'):
        raise NotImplemented

    def topk_recommendation(self, keys, topk) -> dict:
        """Return TopK recommendation for each users(keys)
        """
        if not isinstance(keys, list):
            keys = [keys]
        if not self._idmanager.userid_mapped:
            self.build_userid_map()
        if not self._idmanager.itemid_mapped:
            self.build_itemid_map()
        rows = [self._idmanager.userid_map[k] for k in keys
                if k in self._idmanager.userid_map]
        topks = self._get_topk_recommendation(rows, topk)
        return {self._idmanager.userids[k]: [self._idmanager.itemids[v] for v in vv]
                for k, vv in topks}

    def most_similar(self, key, topk=10, group='item'):
        if group == 'item':
            if not self._idmanager.itemid_mapped:
                self.build_itemid_map()
            return self.most_similar_item(key, topk)
        return []

    def _get_most_similar_item(self, col, topk, Factor, nrz):
        if isinstance(col, np.ndarray):
            q = col
        else:
            topk += 1
            q = Factor[col]
        if nrz:
            dot = q.dot(Factor.T)
            topks = np.argsort(dot)[-topk:][::-1]
        else:
            dot = q.dot(Factor.T)
            dot = dot / (np.linalg.norm(q) * np.linalg.norm(Factor, axis=1))
            topks = np.argsort(dot)[-topk:][::-1]
        scores = dot[topks]
        return topks, scores

    def most_similar_item(self, key, topk=10):
        """Return most similar items for key
        """
        is_vector = False
        if isinstance(key, np.ndarray):
            f = key
            is_vector = True
        else:
            if not self._idmanager.itemid_mapped:
                self.build_itemid_map()
            col = self._idmanager.itemid_map.get(key)
            if not col:
                return []
            f = col
        topks, scores = self._get_most_similar_item(f, topk + 1)
        if is_vector:
            return [(self._idmanager.itemids[k], v)
                    for (k, v) in zip(topks, scores)]
        else:
            return [(self._idmanager.itemids[k], v)
                    for (k, v) in zip(topks, scores)
                    if k != f]

    def build_itemid_map(self):
        idmap = self.data.get_group('idmap')
        header = self.data.get_header()
        if idmap['cols'].shape[0] == 0:
            self._idmanager.itemids = list(map(str, list(range(header['num_items']))))
            self._idmanager.itemid_map = {str(i): i for i in range(header['num_items'])}
        else:
            self._idmanager.itemids = list(map(lambda x: x.decode('utf-8', 'ignore'), idmap['cols'][::]))
            self._idmanager.itemid_map = {v: idx
                                          for idx, v in enumerate(self._idmanager.itemids)}
        self._idmanager.itemid_mapped = True

    def build_userid_map(self):
        idmap = self.data.get_group('idmap')
        header = self.data.get_header()
        if idmap['rows'].shape[0] == 0:
            self._idmanager.userids = list(map(str, list(range(header['num_users']))))
            self._idmanager.userid_map = {str(i): i for i in range(header['num_users'])}
        else:
            self._idmanager.userids = list(map(lambda x: x.decode('utf-8', 'ignore'), idmap['rows'][::]))
            self._idmanager.userid_map = {v: idx
                                          for idx, v in enumerate(self._idmanager.userids)}
        self._idmanager.userid_mapped = True

    def get_feature(self, name, group='item'):
        index = None
        if group == 'item':
            if not self._idmanager.itemid_mapped:
                self.build_itemid_map()
            index = self._idmanager.itemid_map.get(name)
            if not index:
                return None
        elif group == 'user':
            if not self._idmanager.userid_mapped:
                self.build_userid_map()
            index = self._idmanager.userid_map.get(name)
            if not index:
                return None
        return self._get_feature(index, group)

    @abc.abstractmethod
    def _get_feature(self, index, group='item'):
        raise NotImplemented

    def get_weighted_feature(self, weights, group='item', min_length=1):
        if isinstance(weights, dict):
            feat = [(self.get_feature(k), w) for k, w in weights.items()]
            feat = [f * w for f, w in feat if f is not None]
        elif isinstance(weights, list):
            feat = [self.get_feature(k) for k, w in weights]
        if len(feat) < min_length:
            return None
        feat = np.array(feat, dtype=np.float64).mean(axis=0)
        return (feat / np.linalg.norm(feat)).astype(np.float32)


class Serializable(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    def save(self, path, with_itemid_map=True, with_userid_map=True, data_fields=[]):
        if with_itemid_map and not self._idmanager.itemid_mapped:
            self.build_itemid_map()
        if with_userid_map and not self._idmanager.userid_mapped:
            self.build_userid_map()
        data = self._get_data()
        if data_fields:
            data = [(k, v) for k, v in data if k in data_fields]
        with open(path, 'wb') as fout:
            total_objs = len(data)
            fout.write(struct.pack('Q', total_objs))
            for name, obj in data:
                bname = bytes(name, encoding='utf-8')
                fout.write(struct.pack('Q', len(bname)))
                fout.write(bname)
                s = pickle.dumps(obj, protocol=4)
                fout.write(struct.pack('Q', len(s)))
                fout.write(s)

    def _get_data(self):
        data = [('_idmanager', self._idmanager)]
        return data

    def load(self, path, data_fields=[]):
        with open(path, 'rb') as fin:
            total_objs = struct.unpack('Q', fin.read(8))[0]
            for _ in range(total_objs):
                name_sz = struct.unpack('Q', fin.read(8))[0]
                name = fin.read(name_sz).decode('utf8')
                obj_sz = struct.unpack('Q', fin.read(8))[0]
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


class TensorboardExtention(object):
    @abc.abstractmethod
    def get_evaluation_metrics(self):
        raise NotImplemented

    def _get_initial_tensorboard_data(self):
        tb = aux.Option({'summary_writer': None,
                         'name': None,
                         'metrics': {},
                         'feed_dict': {},
                         'merged_summary_op': None,
                         'session': None,
                         'pbar': None,
                         'data_root': None,
                         'step': 1})
        return tb

    def initialize_tensorboard(self, num_steps, name_prefix='', name_postfix='', metrics=None):
        if not self.opt.tensorboard:
            if not hasattr(self, '_tb_setted'):
                self.logger.debug('Cannot find tensorboard configuration.')
            self.tb_setted = False
            return
        name = self.opt.tensorboard.name
        name = name_prefix + name + name_postfix
        dtm = datetime.datetime.now().strftime('%Y%m%d-%H.%M')
        template = self.opt.tensorboard.get('name_template', '{name}.{dtm}')
        self._tb = self._get_initial_tensorboard_data()
        self._tb.name = template.format(name=name, dtm=dtm)
        if not os.path.isdir(self.opt.tensorboard.root):
            os.makedirs(self.opt.tensorboard.root)
        tb_dir = os.path.join(self.opt.tensorboard.root, self._tb.name)
        self._tb.data_root = tb_dir
        self._tb.summary_writer = tf..summary.FileWriter(tb_dir)
        if not metrics:
            metrics = self.get_evaluation_metrics()
        for m in metrics:
            self._tb.metrics[m] = tf.placeholder(tf.float32)
            tf.summary.scalar(m, self._tb.metrics[m])
            self._tb.feed_dict[self._tb.metrics[m]] = 0.0
        self._tb.merged_summary_op = tf..summary.merge_all()
        self._tb.session = tf.Session()
        self._tb.pbar = Progbar(num_steps, stateful_metrics=self._tb.metrics, verbose=0)
        self._tb_setted = True

    def update_tensorboard_data(self, metrics):
        if not self.opt.tensorboard:
            return
        metrics = [(m, np.float32(metrics.get(m, 0.0)))
                   for m in self._tb.metrics.keys()]
        self._tb.feed_dict = {self._tb.metrics[k]: v
                              for k, v in metrics}
        summary = self._tb.session.run(self._tb.merged_summary_op,
                                       feed_dict=self._tb.feed_dict)
        self._tb.summary_writer.add_summary(summary, self._tb.step)
        self._tb.pbar.update(self._tb.step, metrics)
        self._tb.step += 1

    def finalize_tensorboard(self):
        if not self.opt.tensorboard:
            return
        with open(os.path.join(self._tb.data_root, 'opt.json'), 'w') as fout:
            fout.write(json.dumps(self.opt, indent=2))
        self._tb.summary_writer.close()
        self._tb.session.close()
        self._tb = None
        tf.reset_default_graph()

    def __del__(self):
        if hasattr(self, '_tb_setted') and self._tb_setted:
            tf.reset_default_graph()

    def periodical(self, period, current):
        if not period or (current + 1) % period == 0:
            return True
        return False
