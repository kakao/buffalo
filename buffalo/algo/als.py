import json
import time
from typing import Callable, Dict, Optional

import numpy as np

import buffalo.data
from buffalo.algo._als import CyALS
from buffalo.algo.base import Algo, Serializable
from buffalo.algo.options import ALSOption
from buffalo.data.base import Data
from buffalo.data.buffered_data import BufferedDataMatrix
from buffalo.evaluate import Evaluable
from buffalo.misc import aux, log

inited_CUALS = True
try:
    from buffalo.algo.cuda._als import CyALS as CuALS
except Exception:
    inited_CUALS = False


class ALS(Algo, ALSOption, Evaluable, Serializable):
    """Python implementation for C-ALS.

    Implementation of Collaborative Filtering for Implicit Feedback datasets.

    Reference: http://yifanhu.net/PUB/cf.pdf"""
    def __init__(self, opt_path=None, *args, **kwargs):
        Algo.__init__(self, *args, **kwargs)
        ALSOption.__init__(self, *args, **kwargs)
        Evaluable.__init__(self, *args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        if opt_path is None:
            opt_path = ALSOption().get_default_option()

        self.logger = log.get_logger("ALS")
        self.opt, self.opt_path = self.get_option(opt_path)
        if self.opt.accelerator and not inited_CUALS:
            self.logger.error("ImportError CuALS, no cuda library exists.")
            raise RuntimeError()
        self.obj = CuALS() if self.opt.accelerator else CyALS()
        assert self.obj.init(bytes(self.opt_path, "utf-8")), "cannot parse option file: %s" % opt_path

        self.data = None
        data = kwargs.get("data")
        data_opt = self.opt.get("data_opt")
        data_opt = kwargs.get("data_opt", data_opt)
        if data_opt:
            self.data = buffalo.data.load(data_opt)
            self.data.create()
        elif isinstance(data, Data):
            self.data = data
        self.logger.info("ALS(%s)" % json.dumps(self.opt, indent=2))
        if self.data:
            self.logger.info(self.data.show_info())
            assert self.data.data_type in ["matrix"]

    @staticmethod
    def new(path, data_fields=[]):
        return ALS.instantiate(ALSOption, path, data_fields)

    def set_data(self, data):
        assert isinstance(data, aux.data.Data), "Wrong instance: {}".format(type(data))
        self.data = data

    def normalize(self, group="item"):
        if group == "item" and not self.opt._nrz_Q:
            self.Q = self._normalize(self.Q)
            self.opt._nrz_Q = True
        elif group == "user" and not self.opt._nrz_P:
            self.P = self._normalize(self.P)
            self.opt._nrz_P = True

    def initialize(self):
        super().initialize()
        self.init_factors()

    def init_factors(self):
        assert self.data, "Data is not set"
        self.vdim = self.obj.get_vdim() if self.opt.accelerator else self.opt.d
        header = self.data.get_header()
        for name, rows in [("P", header["num_users"]), ("Q", header["num_items"])]:
            setattr(self, name, None)
            setattr(self, name, np.abs(np.random.normal(scale=1.0 / (self.opt.d ** 2),
                                       size=(rows, self.vdim)).astype("float32")))
        self.P[:, self.opt.d:] = 0.0
        self.Q[:, self.opt.d:] = 0.0
        self.obj.initialize_model(self.P, self.Q)

    def _get_topk_recommendation(self, rows, topk, pool=None):
        p = self.P[rows]
        topks = super()._get_topk_recommendation(
            p, self.Q,
            pb=None, Qb=None,
            pool=pool, topk=topk, num_workers=self.opt.num_workers)
        return zip(rows, topks)

    def _get_most_similar_item(self, col, topk, pool):
        return super()._get_most_similar_item(col, topk, self.Q, self.opt._nrz_Q, pool)

    def get_scores(self, row_col_pairs):
        rets = {(r, c): self.P[r].dot(self.Q[c]) for r, c in row_col_pairs}
        return rets

    def _get_scores(self, row, col):
        scores = (self.P[row] * self.Q[col]).sum(axis=1)
        return scores

    def _get_buffer(self):
        buf = BufferedDataMatrix()
        buf.initialize(self.data)
        return buf

    def _iterate(self, buf, group="rowwise"):
        header = self.data.get_header()
        # end = header["num_users"] if group == "rowwise" else header["num_items"]
        int_group = 0 if group == "rowwise" else 1
        st = time.time()
        self.obj.precompute(int_group)
        el, st = time.time() - st, time.time()
        loss_nume, loss_deno = 0.0, 0.0
        update_t, feed_t, updated = el, 0, 0
        buf.set_group(group)
        with log.ProgressBar(log.DEBUG, desc="%s" % group,
                             total=header["num_nnz"], mininterval=30) as pbar:
            for sz in buf.fetch_batch():
                updated += sz
                start_x, next_x, indptr, keys, vals = buf.get()
                _feed_t, st = time.time() - st, time.time()

                _loss_nume, _loss_deno = self.obj.partial_update(start_x, next_x, indptr, keys, vals, int_group)
                loss_nume += _loss_nume
                loss_deno += _loss_deno

                _update_t, st = time.time() - st, time.time()
                pbar.update(sz)
                feed_t += _feed_t
                update_t += _update_t
        self.logger.debug(
            f"{group} updated: processed({updated}) elapsed(data feed: {feed_t:0.3f}s update: {update_t:0.03}s)")
        return loss_nume, loss_deno

    def train(self, training_callback: Optional[Callable[[int, Dict[str, float]], None]] = None):
        if self.opt.accelerator:
            for attr in ["P", "Q"]:
                F = getattr(self, attr)
                if F.shape[1] < self.vdim:
                    _F = np.empty(shape=(F.shape[0], self.vdim), dtype=np.float32)
                    _F[:, :self.P.shape[1]] = F
                    _F[:, self.opt.d:] = 0.0
                    setattr(self, attr, _F)
            self.obj.initialize_model(self.P, self.Q)

        buf = self._get_buffer()
        if self.opt.accelerator:
            lindptr, rindptr, batch_size = buf.get_indptrs()
            self.obj.set_placeholder(lindptr, rindptr, batch_size)

        best_loss, rmse, self.validation_result = float("inf"), None, {}
        full_st = time.time()
        for i in range(self.opt.num_iters):
            start_t = time.time()

            _loss_nume1, _loss_deno1 = self._iterate(buf, group="rowwise")
            _loss_nume2, _loss_deno2 = self._iterate(buf, group="colwise")
            loss_nume = _loss_nume1 + _loss_nume2
            loss_deno = _loss_deno1 + _loss_deno2

            train_t = time.time() - start_t
            rmse = (loss_nume / (loss_deno + self.opt.eps)) ** 0.5
            metrics = {"train_loss": rmse}
            if self.opt.validation and \
               self.opt.evaluation_on_learning and \
               self.periodical(self.opt.evaluation_period, i):
                start_t = time.time()
                self.validation_result = self.get_validation_results()
                vali_t = time.time() - start_t
                val_str = " ".join([f"{k}:{v:0.5f}" for k, v in self.validation_result.items()])
                self.logger.info(f"Validation: {val_str} Elapsed {vali_t:0.3f} secs")
                metrics.update({"val_%s" % k: v
                                for k, v in self.validation_result.items()})
                if training_callback is not None and callable(training_callback):
                    training_callback(i, metrics)
            self.logger.info("Iteration %d: RMSE %.3f Elapsed %.3f secs" % (i + 1, rmse, train_t))
            best_loss = self.save_best_only(rmse, best_loss, i)
            if self.early_stopping(rmse):
                break
        full_el = time.time() - full_st
        self.logger.info(f"elapsed for full epochs: {full_el:.2f} sec")
        if self.opt.accelerator and self.opt.d < self.vdim:
            self.P = self.P[:, :self.opt.d]
            self.Q = self.Q[:, :self.opt.d]
        ret = {"train_loss": rmse}
        ret.update({"val_%s" % k: v
                    for k, v in self.validation_result.items()})
        return ret

    def _get_feature(self, index, group="item"):
        if group == "item":
            return self.Q[index]
        elif group == "user":
            return self.P[index]
        return None

    def _get_data(self):
        data = super()._get_data()
        data.extend([("opt", self.opt),
                     ("Q", self.Q),
                     ("P", self.P)])
        return data

    def get_evaluation_metrics(self):
        return ["train_loss", "val_rmse", "val_ndcg", "val_map", "val_accuracy", "val_error"]
