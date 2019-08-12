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

        # put options into cython class with type assertion
        # see comments on options.py for the description of each parameter
        self.opt, self.opt_path = self.get_option(opt_path)
        self.obj = CyCFR()
        # check the validity of option
        self.is_valid_option(self.opt)
        self.obj.init(self.opt_path)

        # ensure embedding matrix is initialzed for preventing segmentation fault
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
        self.logger.info('CFR ({})'.format(json.dumps(self.opt, indent=2)))
        if self.data:
            self.logger.info(self.data.show_info())
            assert self.data.data_type in ['stream']

    @staticmethod
    def new(path, data_fields=[]):
        return CFR.instantiate(CFROption, path, data_fields)

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
        for attr_name in ['U', 'I', 'C', 'Ib', 'Cb']:
            setattr(self, attr_name, None)
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
        if group == "user":
            self.obj.precompute("item".encode("utf8"))
            total = header["num_nnz"]
            _groups = ["rowwise"]
        elif group == "item":
            self.obj.precompute("user".encode("utf8"))
            total = header["num_nnz"] + header["sppmi_nnz"]
            _groups = ["colwise", "sppmi"]
        elif group == "context":
            total = header["sppmi_nnz"]
            _groups = ["sppmi"]

        with log.pbar(log.DEBUG, desc='%s' % group,
                      total=total, mininterval=30) as pbar:
            start_t = time.time()
            for start_x, next_x in buf.fetch_batch_range(_groups):
                feed_t += time.time() - start_t
                start_t = time.time()
                _err, _updated = self.partial_update(buf, group, start_x, next_x)
                update_t += time.time() - start_t
                updated += _updated
                err += _err
                pbar.update(_updated)
            pbar.refresh()
        self.logger.debug(f'updated {group} processed({updated}) elapsed(data feed: {feed_t:.3f} update: {update_t:.3f}")')
        return err

    def partial_update(self, buf, group, start_x, next_x):
        if group == "user":
            indptr, keys, vals = buf.get_specific_chunk("rowwise", start_x, next_x)
            return self.obj.partial_update_user(start_x, next_x, indptr, keys, vals), len(keys)
        elif group == "item":
            indptr_u, keys_u, vals_u = buf.get_specific_chunk("colwise", start_x, next_x)
            indptr_c, keys_c, vals_c = buf.get_specific_chunk("sppmi", start_x, next_x)
            return self.obj.partial_update_item(start_x, next_x, indptr_u, keys_u, vals_u, indptr_c, keys_c, vals_c), \
                len(keys_u) + len(keys_c)
        elif group == "context":
            indptr, keys, vals = buf.get_specific_chunk("sppmi", start_x, next_x)
            return self.obj.partial_update_context(start_x, next_x, indptr, keys, vals)

    def compute_scale(self):
        # scaling loss for convenience in monitoring
        ret = self.data.get_scale_info(with_sppmi=True)
        num_users, num_items, sppmi_nnz, vsum = \
            [ret[k] for k in ["num_users", "num_items", "sppmi_nnz", "vsum"]]
        alpha, l = self.opt.alpha, self.opt.l
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
