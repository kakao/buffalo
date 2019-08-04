# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
# -*- coding: utf-8 -*-
import time
import json
import logging

import tqdm
import cython
from libcpp cimport bool
from libc.stdint cimport int64_t
from libcpp.string cimport string
from eigency.core cimport MatrixXf, Map, VectorXi, VectorXf

import numpy as np
cimport numpy as np
from numpy.linalg import norm
from hyperopt import STATUS_OK as HOPT_STATUS_OK

import buffalo.data
from buffalo.data.base import Data
from buffalo.misc import aux, log
from buffalo.evaluate import Evaluable
from buffalo.algo.options import BprmfOption
from buffalo.algo.optimize import Optimizable
from buffalo.data.buffered_data import BufferedDataMatrix
from buffalo.algo.base import Algo, Serializable, TensorboardExtention


cdef extern from "buffalo/algo_impl/bpr/bpr.hpp" namespace "bpr":
    cdef cppclass CBPRMF:
        void release() nogil except +
        bool init(string) nogil except +
        void set_factors(Map[MatrixXf]&,
                         Map[MatrixXf]&,
                         Map[MatrixXf]&) nogil except +
        void set_cumulative_table(int64_t*, int)
        double partial_update(int,
                              int,
                              int64_t*,
                              Map[VectorXi]&) nogil except +
        double compute_loss(Map[VectorXi]&,
                            Map[VectorXi]&,
                            Map[VectorXi]&) nogil except +
        void update_parameters()


cdef class PyBPRMF:
    """CBPRMF object holder"""
    cdef CBPRMF* obj  # C-BPRMF object

    def __cinit__(self):
        self.obj = new CBPRMF()

    def __dealloc__(self):
        self.obj.release()
        del self.obj

    def init(self, option_path):
        return self.obj.init(option_path)

    def set_factors(self,
                    np.ndarray[np.float32_t, ndim=2] P,
                    np.ndarray[np.float32_t, ndim=2] Q,
                    np.ndarray[np.float32_t, ndim=2] Qb):
        self.obj.set_factors(Map[MatrixXf](P),
                             Map[MatrixXf](Q),
                             Map[MatrixXf](Qb))

    def set_cumulative_table(self,
                             np.ndarray[np.int64_t, ndim=1] cum_table,
                             int cum_table_size):
        self.obj.set_cumulative_table(&cum_table[0], cum_table_size)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def partial_update(self,
                       int start_x,
                       int next_x,
                       np.ndarray[np.int64_t, ndim=1] indptr,
                       np.ndarray[np.int32_t, ndim=1] positives):
        return self.obj.partial_update(start_x,
                                       next_x,
                                       &indptr[0],
                                       Map[VectorXi](positives))

    def compute_loss(self,
                     np.ndarray[np.int32_t, ndim=1] users,
                     np.ndarray[np.int32_t, ndim=1] positives,
                     np.ndarray[np.int32_t, ndim=1] negatives):
        return self.obj.compute_loss(Map[VectorXi](users),
                                     Map[VectorXi](positives),
                                     Map[VectorXi](negatives))

    def update_parameters(self):
        self.obj.update_parameters()


class BPRMF(Algo, BprmfOption, Evaluable, Serializable, Optimizable, TensorboardExtention):
    """Python implementation for C-BPRMF.
    """
    def __init__(self, opt_path, *args, **kwargs):
        Algo.__init__(self, *args, **kwargs)
        BprmfOption.__init__(self, *args, **kwargs)
        Evaluable.__init__(self, *args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        Optimizable.__init__(self, *args, **kwargs)

        self.logger = log.get_logger('BPRMF')
        self.opt, self.opt_path = self.get_option(opt_path)
        self.obj = PyBPRMF()
        assert self.obj.init(bytes(self.opt_path, 'utf-8')),\
            'cannot parse option file: %s' % opt_path
        self.data = None
        data = kwargs.get('data')
        data_opt = self.opt.get('data_opt')
        data_opt = kwargs.get('data_opt', data_opt)
        if data_opt:
            self.data = buffalo.data.load(data_opt)
            self.data.create()
        elif isinstance(data, Data):
            self.data = data
        self.logger.info('BPRMF(%s)' % json.dumps(self.opt, indent=2))
        if self.data:
            self.logger.info(self.data.show_info())
            assert self.data.data_type in ['matrix']

    def set_data(self, data):
        assert isinstance(data, aux.data.Data), 'Wrong instance: {}'.format(type(data))
        self.data = data

    def fast_similar(self, onoff=True):
        # TODO: implement
        if onoff:
            self.FQ = self.normalize(self.Q)
        else:
            if hasattr(self, 'FQ'):
                del self.FQ

    def normalize(self, group='item'):
        if group == 'item':
            self.Q = self._normalize(self.Q)
            self.opt._nrz_Q = True
        elif group == 'user':
            self.P = self._normalize(self.P)
            self.opt._nrz_P = True

    def initialize(self):
        assert self.data, 'Data is not setted'
        if self.opt.random_seed:
            np.random.seed(self.opt.random_seed)
        self.buf = BufferedDataMatrix()
        self.buf.initialize(self.data)
        self.init_factors()
        self.prepare_sampling()

    def init_factors(self):
        header = self.data.get_header()
        self.P = np.abs(np.random.normal(scale=1.0/(self.opt.d ** 2),
                                         size=(header['num_users'], self.opt.d)).astype("float32"), order='F')
        self.Q = np.abs(np.random.normal(scale=1.0/(self.opt.d ** 2),
                                         size=(header['num_items'], self.opt.d)).astype("float32"), order='F')
        self.Qb = np.abs(np.random.normal(scale=1.0/(self.opt.d ** 2),
                                          size=(header['num_items'], 1)).astype("float32"), order='F')
        self.obj.set_factors(self.P, self.Q, self.Qb)

    def prepare_sampling(self):
        self.logger.info('Preparing sampling ...')
        header = self.data.get_header()
        self.sampling_table_ = np.zeros(header['num_items'], dtype=np.int64)
        if self.opt.sampling_power > 0.0:
            for sz in self.buf.fetch_batch():
                *_, keys, vals = self.buf.get()
                for i in range(sz):
                    self.sampling_table_[keys[i]] += 1
            self.sampling_table_ **= int(self.opt.sampling_power)
            for i in range(1, header['num_items']):
                self.sampling_table_[i] += self.sampling_table_[i - 1]
        else:
            for i in range(1, header['num_items'] + 1):
                self.sampling_table_[i - 1] = i
        self.obj.set_cumulative_table(self.sampling_table_, header['num_items'])

    def _get_topk_recommendation(self, rows, topk):
        p = self.P[rows]
        topks = np.argsort(p.dot(self.Q.T) + self.Qb.T, axis=1)[:, -topk:][:,::-1]
        return zip(rows, topks)

    def _get_most_similar_item(self, col, topk):
        return super()._get_most_similar_item(col, topk, self.Q, self.opt._nrz_Q)

    def get_scores(self, row_col_pairs):
        rets = {(r, c): self.P[r].dot(self.Q[c]) + self.Qb[c][0] for r, c in row_col_pairs}
        return rets

    def sampling_loss_samples(self):
        header = self.data.get_header()
        num_loss_samples = int(header['num_users'] ** 0.5)
        _users = np.random.choice(range(self.P.shape[0]), size=num_loss_samples, replace=False)
        users = []
        positives, negatives = [], []
        for u in _users:
            keys, *_ = self.data.get(u)
            if len(keys) == 0:
                continue
            seen = set(keys)
            negs = np.random.choice(range(self.Q.shape[0]),
                                    size=len(seen) + 1,
                                    replace=False)
            negs = [n for n in negs if n not in seen]
            users.append(u)
            positives.append(keys[0])
            negatives.append(negs[0])
        self._sub_samples = [
            np.array(users, dtype=np.int32, order='F'),
            np.array(positives, dtype=np.int32, order='F'),
            np.array(negatives, dtype=np.int32, order='F')
        ]
        self.logger.info('Generated %s loss samples.' % len(users))

    def _get_feature(self, index, group='item'):
        if group == 'item':
            return self.Q[index]
        elif group == 'user':
            return self.P[index]
        return None

    def _iterate(self):
        header = self.data.get_header()
        end = header['num_users']
        update_t, feed_t, updated = 0, 0, 0
        self.buf.set_group('rowwise')
        with log.pbar(log.DEBUG,
                      total=header['num_nnz'], mininterval=30) as pbar:
            start_t = time.time()
            for sz in self.buf.fetch_batch():
                updated += sz
                feed_t += time.time() - start_t
                start_x, next_x, indptr, keys, vals = self.buf.get()

                start_t = time.time()
                self.obj.partial_update(start_x, next_x, indptr, keys)
                update_t += time.time() - start_t
                pbar.update(sz)
            pbar.refresh()
        self.obj.update_parameters()
        self.logger.debug(f'updated processed({updated}) elapsed(data feed: {feed_t:0.3f} update: {update_t:0.3f}')

    def compute_loss(self):
        return self.obj.compute_loss(self._sub_samples[0],
                                     self._sub_samples[1],
                                     self._sub_samples[2])

    def train(self):
        rmse, self.validation_result = None, {}
        self.prepare_evaluation()
        self.initialize_tensorboard(self.opt.num_iters)
        self.sampling_loss_samples()
        best_loss = 987654321.0
        for i in range(self.opt.num_iters):
            start_t = time.time()
            self._iterate()
            loss = self.compute_loss()

            self.logger.info('Iteration %s: PR-Loss %.3f Elapsed %.3f secs' % (i + 1, loss, time.time() - start_t))
            metrics = {'prloss': loss}
            if self.opt.validation and self.opt.evaluation_on_learning:
                self.validation_result = self.get_validation_results()
                self.logger.info('Validation: ' + \
                                 ' '.join([f'{k}:{v:0.5f}'
                                           for k, v in self.validation_result.items()]))
                metrics.update({'val_%s' % k: v
                                for k, v in self.validation_result.items()})
            self.update_tensorboard_data(metrics)

            if self.opt.save_best and best_loss > loss and (not self.opt.period or (i + 1) % self.opt.period == 0):
                best_loss = loss
                self.save(self.model_path)
        ret = {'prloss': loss}
        ret.update({'val_%s' % k: v
                    for k, v in self.validation_result.items()})
        self.finalize_tensorboard()
        return ret

    def _optimize(self, params):
        # TODO: implement
        self._optimize_params = params
        for name, value in params.items():
            assert name in self.opt, 'Unexepcted parameter: {}'.format(name)
            setattr(self.opt, name, value)
        with open(self._temporary_opt_file, 'w') as fout:
            json.dump(self.opt, fout, indent=2)
        assert self.obj.init(bytes(self._temporary_opt_file, 'utf-8')),\
            'cannot parse option file: %s' % self._temporary_opt_file
        self.logger.info(params)
        self.init_factors()
        loss = self.train()
        loss['eval_time'] = time.time()
        loss['loss'] = loss.get(self.opt.optimize.loss)
        # TODO: deal with failture of training
        loss['status'] = HOPT_STATUS_OK
        self._optimize_loss = loss
        return loss

    def _get_data(self):
        data = super()._get_data()
        data.extend([('opt', self.opt),
                     ('Q', self.Q),
                     ('Qb', self.Qb),
                     ('P', self.P)])
        return data

    def get_evaluation_metrics(self):
        return ['val_rmse', 'val_ndcg', 'val_map', 'val_accuracy', 'val_error', 'prloss']
