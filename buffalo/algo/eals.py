import json
import time
from typing import Callable, Dict, Optional

import numpy as np

import buffalo.data
from buffalo.algo._eals import CyEALS
from buffalo.algo.base import Algo, Serializable
from buffalo.algo.options import EALSOption
from buffalo.data.base import Data
from buffalo.evaluate import Evaluable
from buffalo.misc import aux, log


class EALS(Algo, EALSOption, Evaluable, Serializable):
    """Python implementation for C-EALS.

    Implementation of Fast Matrix Factorization for Online Recommendation.

    Reference: https://arxiv.org/abs/1708.05024"""
    def __init__(self, opt_path=None, *args, **kwargs):
        Algo.__init__(self, *args, **kwargs)
        EALSOption.__init__(self, *args, **kwargs)
        Evaluable.__init__(self, *args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        if opt_path is None:
            opt_path = EALSOption().get_default_option()

        self.logger = log.get_logger("EALS")
        self.group2axis = {"rowwise": 0, "colwise": 1}
        self.opt, self.opt_path = self.get_option(opt_path)
        self.obj = CyEALS()
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
        self.logger.info("eALS(%s)" % json.dumps(self.opt, indent=2))
        if self.data:
            self.logger.info(self.data.show_info())
            assert self.data.data_type in ["matrix"]

    @staticmethod
    def new(path, data_fields=[]):
        return EALS.instantiate(EALSOption, path, data_fields)

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
        self.vdim = self.opt.d
        header = self.data.get_header()
        self._nnz = header["num_nnz"]
        for name, rows in [("P", header["num_users"]), ("Q", header["num_items"])]:
            setattr(self, name, None)
            setattr(self, name, np.random.normal(scale=1.0 / (self.opt.d ** 2),
                    size=(rows, self.vdim)).astype("float32"))
        self.P[:, self.opt.d:] = 0.0
        self.Q[:, self.opt.d:] = 0.0
        self.C = self._get_negative_weights()

        self.obj.initialize_model(self.P, self.Q, self.C)

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

    def _get_negative_weights(self):
        # Get item popularity from self.data
        indptr, _, _ = self._get_mm_data(group="colwise")
        pop = np.array([indptr[i] - (0 if i == 0 else indptr[i - 1]) for i in range(len(indptr))], dtype="float32")
        assert len(pop) == self.data.get_header()["num_items"]
        # Return negative weights calculated by the power-law weighting scheme
        pop /= max(pop)
        pop_with_exponent = pop**self.opt.get("exponent", 0.0)
        return self.opt.get("c0", 1.0) * pop_with_exponent / sum(pop_with_exponent)

    def _get_mm_data(self, group):
        group = self.data.get_group(group)
        indptr = group["indptr"][:]
        keys = group["key"][:]
        vals = group["val"][:]
        return indptr, keys, vals

    def _precompute_cache(self):
        for group in ["rowwise", "colwise"]:
            indptr, keys, _ = self._get_mm_data(group)
            axis = self.group2axis[group]
            self.obj.precompute_cache(self._nnz, indptr, keys, axis)

    def _iterate(self, group):
        indptr, keys, vals = self._get_mm_data(group)
        axis = self.group2axis[group]
        assert self.obj.update(indptr, keys, vals, axis)

    def _get_loss(self):
        axis = self.group2axis[(group := "rowwise")]
        indptr, keys, vals = self._get_mm_data(group=group)
        # loss: RMSE / total_loss := RMSE^2 + L2-loss + Negative Feedbacks
        loss, total_loss = self.obj.estimate_loss(self._nnz, indptr, keys, vals, axis)
        return loss, total_loss

    def train(self, training_callback: Optional[Callable[[int, Dict[str, float]], None]] = None):
        best_loss, loss, self.validation_result = float("inf"), None, {}
        full_st = time.time()

        self._precompute_cache()

        for i in range(self.opt.num_iters):
            start_t = time.time()
            self._iterate(group="rowwise")
            self._iterate(group="colwise")
            loss, total_loss = self._get_loss()

            train_t = time.time() - start_t
            metrics = {"train_loss": loss}
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
            self.logger.info("Iteration %d: RMSE %.3f TotalLoss %.3f Elapsed %.3f secs" % (i + 1, loss, (total_loss / self._nnz), train_t))
            best_loss = self.save_best_only(loss, best_loss, i)
            if self.early_stopping(loss):
                break

        full_el = time.time() - full_st
        self.logger.info(f"elapsed for full epochs: {full_el:.2f} sec")
        ret = {"train_loss": loss}
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
