import json
import time
from typing import Callable, Dict, Optional

import numpy as np

import buffalo.data
from buffalo.algo._plsi import CyPLSI
from buffalo.algo.base import Algo, Serializable
from buffalo.algo.options import PLSIOption
from buffalo.data.base import Data
from buffalo.data.buffered_data import BufferedDataMatrix
from buffalo.evaluate import Evaluable
from buffalo.misc import aux, log


class PLSI(Algo, PLSIOption, Evaluable, Serializable):
    """Python implementation for pLSI."""
    def __init__(self, opt_path=None, *args, **kwargs):
        Algo.__init__(self, *args, **kwargs)
        PLSIOption.__init__(self, *args, **kwargs)
        Evaluable.__init__(self, *args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        if opt_path is None:
            opt_path = PLSIOption().get_default_option()

        self.logger = log.get_logger("PLSI")
        self.opt, self.opt_path = self.get_option(opt_path)

        self.obj = CyPLSI()
        assert self.obj.init(self.opt_path.encode("utf8")), "putting parameter to cython object failed"

        self.data = None
        data = kwargs.get("data")
        data_opt = self.opt.get("data_opt")
        data_opt = kwargs.get("data_opt", data_opt)
        if data_opt:
            self.data = buffalo.data.load(data_opt)
            assert self.data.data_type == "matrix"
            self.data.create()
        elif isinstance(data, Data):
            self.data = data
        self.logger.info("PLSI ({})".format(json.dumps(self.opt, indent=2)))
        if self.data:
            self.logger.info(self.data.show_info())
            assert self.data.data_type in ["matrix"]

    @staticmethod
    def new(path, data_fields=[]):
        return PLSI.instantiate(PLSIOption, path, data_fields)

    def set_data(self, data):
        assert isinstance(data, aux.data.Data), "Wrong instance: {}".format(type(data))
        self.data = data

    def normalize(self, group="item"):
        if group == "item":
            self.Q /= (np.sum(self.Q, axis=0, keepdims=True) + self.opt.eps)
        elif group == "user":
            self.P /= (np.sum(self.P, axis=1, keepdims=True) + self.opt.eps)

    def inherit(self):
        def _inherit(key):
            if key == "user":
                self.build_userid_map()
            else:
                self.build_itemid_map()
            curr_idmap = self._idmanager.userid_map if key == "user" else self._idmanager.itemid_map
            prev_idmap = prev_model._idmanager.userid_map if key == "user" else prev_model._idmanager.itemid_map
            curr_obj = self.P if key == "user" else self.Q
            prev_obj = prev_model.P if key == "user" else prev_model.Q
            curr_d, prev_d = curr_obj.shape[1], prev_obj.shape[1]
            assert curr_d == prev_d, f"Dimension mismatch. Current dimension: {curr_d} / Previous dimension: {prev_d}"
            for key, curr_idx in curr_idmap.items():
                if key in prev_idmap:
                    prev_idx = prev_idmap[key]
                    curr_obj[curr_idx] = prev_obj[prev_idx]

        if not self.opt["inherit_opt"]:
            return
        inherit_opt = self.opt.inherit_opt
        prev_model = PLSI.new(inherit_opt.model_path)
        if inherit_opt.get("inherit_user", False):
            self.logger.info("Inherit from previous user matrix")
            _inherit("user")

        if inherit_opt.get("inherit_item", False):
            self.logger.info("Inherit from previous item matrix")
            _inherit("item")

    def initialize(self):
        super().initialize()
        self.buf = BufferedDataMatrix()
        self.buf.initialize(self.data)
        self.buf.set_group("rowwise")
        self.init_factors()
        self.inherit()

    def init_factors(self):
        assert self.data, "Did not set data"

        header = self.data.get_header()
        self.num_items = header["num_items"]
        self.num_users = header["num_users"]
        self.num_nnz = header["num_nnz"]

        for name, rows in [("P", self.num_users), ("Q", self.num_items)]:
            setattr(self, name, None)
            setattr(self, name, np.zeros((rows, self.opt.d), dtype="float32"))

        self.obj.initialize_model(self.P, self.Q)

    def _get_topk_recommendation(self, rows, topk, pool=None):
        p = self.P[rows]
        topks = super()._get_topk_recommendation(
            p, self.Q,
            pb=None, Qb=None,
            pool=pool, topk=topk, num_workers=self.opt.num_workers)
        return zip(rows, topks)

    def _get_most_similar_item(self, col, topk, pool):
        return super()._get_most_similar_item(col, topk, self.Q, True, pool)

    def get_scores(self, row_col_pairs):
        rets = {(r, c): self.P[r].dot(self.Q[c]) for r, c in row_col_pairs}
        return rets

    def _get_scores(self, row, col):
        scores = (self.P[row] * self.Q[col]).sum(axis=1)
        return scores

    def _iterate(self):
        self.obj.reset()

        loss_nume, loss_deno = 0.0, 0.0
        update_t, feed_t, updated = 0, 0, 0

        with log.ProgressBar(log.DEBUG, total=self.num_nnz, mininterval=30) as pbar:
            for sz in self.buf.fetch_batch():
                st = time.time()
                start_x, next_x, indptr, keys, vals = self.buf.get()
                feed_t += time.time() - st

                st = time.time()
                _loss = self.obj.partial_update(start_x, next_x, indptr, keys, vals)
                update_t += time.time() - st

                loss_deno += np.sum(vals)
                loss_nume += _loss

                pbar.update(sz)
                updated += sz
            pbar.refresh()

        self.obj.normalize(self.opt.alpha1, self.opt.alpha2)
        self.obj.swap()

        self.logger.debug(
            f"updated processed({updated}) elapsed(data feed: {feed_t:0.5f} update: {update_t:0.5f})")
        return loss_nume, loss_deno

    def train(self, training_callback: Optional[Callable[[int, Dict[str, float]], None]] = None):
        best_loss, loss, self.validation_result = 1e+10, None, {}
        self.logger.info(f"Train pLSI, K: {self.opt.d}, alpha1: {self.opt.alpha1}, "
                         f"alpha2: {self.opt.alpha2}, num_workers: {self.opt.num_workers}")
        for i in range(self.opt.num_iters):
            start_t = time.time()
            _loss_nume, _loss_deno = self._iterate()
            train_t = time.time() - start_t

            loss = _loss_nume / (_loss_deno + self.opt.eps)
            metrics = {"train_loss": loss}

            if self.opt.validation and \
               self.opt.evaluation_on_learning and \
               self.periodical(self.opt.evaluation_period, i):
                start_t = time.time()
                self.validation_result = self.get_validation_results()
                vali_t = time.time() - start_t
                val_str = " ".join([f"{k}:{v:0.5f}" for k, v in self.validation_result.items()])
                self.logger.info(f"Validation: {val_str} Elapsed {vali_t:0.3f} secs")
                metrics.update({"val_%s" % k: v for k, v in self.validation_result.items()})
                if training_callback is not None and callable(training_callback):
                    training_callback(i, metrics)
            self.logger.info("Iteration %d: Loss %.3f Elapsed %.3f secs" % (i + 1, loss, train_t))
            best_loss = self.save_best_only(loss, best_loss, i)
            if self.early_stopping(loss):
                break

        ret = {"train_loss": loss}
        ret.update({"val_%s" % k: v for k, v in self.validation_result.items()})
        return ret

    def _get_feature(self, index, group="item"):
        if group == "item":
            return self.Q[index]
        elif group == "user":
            return self.P[index]
        return None

    def _get_data(self):
        data = super()._get_data()
        data.extend([("opt", self.opt), ("Q", self.Q), ("P", self.P)])
        return data

    def get_evaluation_metrics(self):
        return ["train_loss", "val_rmse", "val_ndcg", "val_map", "val_accuracy", "val_error"]
