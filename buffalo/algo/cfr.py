# -*- coding: utf-8 -*-
import time
import json
import logging

import tqdm
import numpy as np
from hyperopt import STATUS_OK as HOPT_STATUS_OK

import buffalo.data
from buffalo.misc import aux, log
from buffalo.data.base import Data
from buffalo.algo._cfr import CyCFR
from buffalo.evaluate import Evaluable
from buffalo.algo.options import CFROption
from buffalo.algo.optimize import Optimizable
from buffalo.data.buffered_data import BufferedDataMatrix
from buffalo.algo.base import Algo, Serializable, TensorboardExtention


class CFR(Algo, CFROption, Evaluable, Serializable, Optimizable, TensorboardExtention):
    """Python implementation for CoFactor.

    Reference: Factorization Meets the Item Embedding:
        Regularizing Matrix Factorization with Item Co-occurrence

    Paper link: http://dawenl.github.io/publications/LiangACB16-cofactor.pdf
    """
    def __init__(self, opt_path, *args, **kwargs):
        Algo.__init__(self, *args, **kwargs)
        CFROption.__init__(self, *args, **kwargs)
        Evaluable.__init__(self, *args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        Optimizable.__init__(self, *args, **kwargs)

        self.logger = log.get_logger('CFR')

        self.opt, self.opt_path = self.get_option(opt_path)
        # put options into cython class with type assertion
        for k, dt in [("dim", int), ("num_workers", int), ("num_cg_max_iters", int),
                      ("alpha", float), ("l", float), ("cg_tolerance", float),
                      ("reg_u", float), ("reg_i", float), ("reg_c", float),
                      ("compute_loss", bool), ("optimizer", str)]:
            assert isinstance(self.opt.get(k), dt), f"{k} should be {dt} type"
        assert self.opt.optimizer in ["llt", "ldlt", "manual_cgd", "eigen_cgd"], \
            f"optimizer ({self.opt.optimizer}) is not properly provided"

        self.obj = CyCFR(self.opt.dim, self.opt.num_workers, self.opt.num_cg_max_iters,
                         self.opt.alpha, self.opt.l, self.opt.cg_tolerance,
                         self.opt.reg_u, self.opt.reg_i, self.opt.reg_c,
                         self.opt.compute_loss, self.opt.optimizer.encode("utf8"))
        self.is_initialized = False

        self.data = None
        data = kwargs.get('data')
        data_opt = self.opt.get('data_opt')
        data_opt = kwargs.get('data_opt', data_opt)
        if data_opt:
            self.data = buffalo.data.load(data_opt)
            self.data.create()
        elif isinstance(data, Data):
            self.data = data
        self.logger.info('CFR (%s)' % json.dumps(self.opt, indent=2))
        if self.data:
            self.logger.info(self.data.show_info())
            assert self.data.data_type in ['stream']

    @staticmethod
    def new(path, data_fields=[]):
        return CFR.instantiate(AlsOption, path, data_fields)

    def set_data(self, data):
        assert isinstance(data, aux.data.Data), 'Wrong instance: {}'.format(type(data))
        self.data = data

    def normalize(self, group='item'):
        assert group in ["user", "item", "context"], \
            f"group ({group}) is not properly provided"
        if group == 'user':
            self.U = self._normalize(self.U)
            self.opt._nrz_U = True
        elif group == 'item':
            self.I = self._normalize(self.I)
            self.opt._nrz_I = True
        elif group == 'context':
            self.C = self._normalize(self.C)
            self.opt._nrz_C = True

    def initialize(self):
        assert self.data, 'Data is not setted'
        header = self.data.get_header()
        self.U = np.random.normal(scale=1.0/(self.opt.dim ** 2),
                                  size=(header['num_users'], self.opt.dim)).astype("float32")
        self.I = np.random.normal(scale=1.0/(self.opt.dim ** 2),
                                  size=(header['num_items'], self.opt.dim)).astype("float32")
        self.C = np.random.normal(scale=1.0/(self.opt.dim ** 2),
                                  size=(header['num_items'], self.opt.dim)).astype("float32")
        self.Ib = np.random.normal(scale=1.0/(self.opt.dim ** 2),
                                   size=(header['num_items'], 1)).astype("float32")
        self.Cb = np.random.normal(scale=1.0/(self.opt.dim ** 2),
                                   size=(header['num_items'], 1)).astype("float32")
        self.obj.set_embedding(self.U, "user".encode("utf8"))
        self.obj.set_embedding(self.I, "item".encode("utf8"))
        self.obj.set_embedding(self.C, "context".encode("utf8"))
        self.obj.set_embedding(self.Ib, "item_bias".encode("utf8"))
        self.obj.set_embedding(self.Cb, "context_bias".encode("utf8"))
        self.is_initialized = True

    def _get_topk_recommendation(self, rows, topk):
        u = self.U[rows]
        topks = np.argsort(u.dot(self.I.T), axis=1)[:, -topk:][:,::-1]
        return zip(rows, topks)

    def _get_most_similar_item(self, col, topk):
        return super()._get_most_similar_item(col, topk, self.I, self.opt._nrz_I)

    def get_scores(self, row_col_pairs):
        rets = {(r, c): self.U[r].dot(self.I[c]) for r, c in row_col_pairs}
        return rets

    def _get_buffer(self):
        buf = BufferedDataMatrix()
        buf.initialize(self.data, order='C', with_sppmi=True)
        return buf

    def _iterate(self, buf, group='user'):
        assert group in ["user", "item", "context"], f"group {group} is not properly provided"
        header = self.data.get_header()
        end = header['num_users'] if group == 'user' else header['num_items']
        err = 0.0
        update_t, feed_t, updated = 0, 0, 0
        buf_map = {"user": "rowwise", "item": "colwise", "context": "sppmi"}
        buf.set_group(buf_map[group])
        total = self.data.handle.attrs["sppmi_nnz"] if group == "context" else header["num_nnz"]
        if group == "user":
            self.obj.precompute("item".encode("utf8"))
        elif group == "item":
            self.obj.precompute("user".encode("utf8"))
        with log.pbar(log.DEBUG, desc='%s' % group,
                      total=total, mininterval=30) as pbar:
            start_t = time.time()
            for sz in buf.fetch_batch():
                updated += sz
                feed_t += time.time() - start_t
                start_x, next_x, indptr, keys, vals = buf.get()
                start_t = time.time()
                _indptr = np.empty(next_x - start_x + 1, dtype=np.int64)
                _indptr[0] = 0 if start_x == 0 else indptr[start_x - 1]
                _indptr[1:] = indptr[start_x: next_x]
                err += self.partial_update(group, start_x, next_x,
                                           _indptr, keys, vals)
                update_t += time.time() - start_t
                pbar.update(sz)
            pbar.refresh()
        self.logger.debug('updated %s processed(%s) elapsed(data feed: %.3f update: %.3f)' % (group, updated, feed_t, update_t))
        return err

    def partial_update(self, group, start_x, next_x, indptr, keys, vals):
        if group == "user":
            return self.obj.partial_update_user(start_x, next_x, indptr, keys, vals)
        elif group == "item":
            db = self.data.get_group("sppmi")
            _indptr = np.empty(next_x - start_x + 1, dtype=np.int64)
            _indptr[0] = 0 if start_x == 0 else db['inpdtr'][start_x - 1]
            _indptr[1:] = db['indptr'][start_x: next_x]
            _keys = db['key'][_indptr[0]: _indptr[-1]]
            _vals = db['val'][_indptr[0]: _indptr[-1]]
            return self.obj.partial_update_item(start_x, next_x, indptr, keys, vals,
                                                _indptr, _keys, _vals)
        elif group == "context":
            return self.obj.partial_update_context(start_x, next_x, indptr, keys, vals)

    def compute_scale(self):
        # scaling loss for convenience in monitoring
        handle = self.data.handle
        num_users, num_items, num_nnz, sppmi_nnz = \
            [handle.attrs[k] for k in ["num_users", "num_items", "num_nnz", "sppmi_nnz"]]
        alpha = self.opt.alpha
        l = self.opt.l
        chunk_size = 100000
        vsum = 0
        db = self.data.get_group("rowwise")
        for offset in range(0, num_nnz, chunk_size):
            limit = min(num_nnz, offset + chunk_size)
            vsum += np.sum(db["val"][offset: limit])
        return l * (alpha * vsum + num_users * num_items) + sppmi_nnz

    def train(self):
        assert self.is_initialized, "embedding matrix is not initialized"
        buf = self._get_buffer()
        best_loss, rmse, self.validation_result = 987654321.0, None, {}
        self.prepare_evaluation()
        self.initialize_tensorboard(self.opt.num_iters)
        scale = self.compute_scale()
        for i in range(self.opt.num_iters):
            start_t = time.time()
            loss = self._iterate(buf, group='user')
            loss += self._iterate(buf, group='item')
            loss += self._iterate(buf, group='context')
            loss /= scale
            train_t = time.time() - start_t
            metrics = {'train_loss': loss}
            if self.opt.validation and self.opt.evaluation_on_learning and self.periodical(self.opt.evaluation_period, i):
                start_t = time.time()
                self.validation_result = self.get_validation_results()
                vali_t = time.time() - start_t
                val_str = ' '.join([f'{k}:{v:0.5f}' for k, v in self.validation_result.items()])
                self.logger.info(f'Validation: {val_str} Elased {vali_t:0.3f}')
                metrics.update({'vali_%s' % k: v
                                for k, v in self.validation_result.items()})
            self.logger.info('Iteration %d: Loss %.3f Elapsed %.3f secs' % (i + 1, loss, train_t))
            self.update_tensorboard_data(metrics)
            if self.opt.save_best and best_loss > loss and self.periodical(self.opt.save_period, i):
                best_loss = loss
                self.save(self.model_path)
        ret = {'train_loss': loss}
        ret.update({'vali_%s' % k: v
                    for k, v in self.validation_result.items()})
        self.finalize_tensorboard()
        return ret

    def _optimize(self, params):
        self._optimize_params = params
        for name, value in params.items():
            assert name in self.opt, 'Unexepcted parameter: {}'.format(name)
            setattr(self.opt, name, value)
        with open(self._temporary_opt_file, 'w') as fout:
            json.dump(self.opt, fout, indent=2)
        assert self.obj.init(bytes(self._temporary_opt_file, 'utf-8')),\
            'cannot parse option file: %s' % self._temporary_opt_file
        self.logger.info(params)
        self.initialize()
        loss = self.train()
        loss['eval_time'] = time.time()
        loss['loss'] = loss.get(self.opt.optimize.loss)
        # TODO: deal with failture of training
        loss['status'] = HOPT_STATUS_OK
        self._optimize_loss = loss
        return loss

    def _get_feature(self, index, group='item'):
        if group == 'item':
            return self.I[index]
        elif group == 'user':
            return self.U[index]
        elif group == 'context':
            return self.C[index]
        return None

    def _get_data(self):
        data = super()._get_data()
        data.extend([('opt', self.opt),
                     ('I', self.I),
                     ('U', self.U),
                     ('C', self.C)])
        return data

    def get_evaluation_metrics(self):
        return ['train_loss', 'vali_rmse', 'vali_ndcg', 'vali_map', 'vali_accuracy', 'vali_error']
