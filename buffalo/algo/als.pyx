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
from buffalo.algo.options import AlsOption
from buffalo.algo.optimize import Optimizable
from buffalo.data.buffered_data import BufferedDataMatrix
from buffalo.algo.base import Algo, Serializable, TensorboardExtention


cdef extern from "buffalo/algo_impl/als/als.hpp" namespace "als":
    cdef cppclass CALS:
        void release() nogil except +
        bool init(string) nogil except +
        void set_factors(Map[MatrixXf]&,
                         Map[MatrixXf]&) nogil except +
        void precompute(int) nogil except +
        double partial_update(int,
                              int,
                              int64_t*,
                              Map[VectorXi]&,
                              Map[VectorXf]&,
                              int) nogil except +


cdef class PyALS:
    """CALS object holder"""
    cdef CALS* obj  # C-ALS object

    def __cinit__(self):
        self.obj = new CALS()

    def __dealloc__(self):
        self.obj.release()
        del self.obj

    def init(self, option_path):
        return self.obj.init(option_path)

    def set_factors(self,
                    np.ndarray[np.float32_t, ndim=2] P,
                    np.ndarray[np.float32_t, ndim=2] Q):
        self.obj.set_factors(Map[MatrixXf](P),
                             Map[MatrixXf](Q))

    def precompute(self, axis):
        self.obj.precompute(axis)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def partial_update(self,
                       int start_x,
                       int next_x,
                       np.ndarray[np.int64_t, ndim=1] indptr,
                       np.ndarray[np.int32_t, ndim=1] keys,
                       np.ndarray[np.float32_t, ndim=1] vals,
                       int axis):
        return self.obj.partial_update(start_x,
                                       next_x,
                                       &indptr[0],
                                       Map[VectorXi](keys),
                                       Map[VectorXf](vals),
                                       axis)


class ALS(Algo, AlsOption, Evaluable, Serializable, Optimizable, TensorboardExtention):
    """Python implementation for C-ALS.

    Implementation of Collaborative Filtering for Implicit Feedback datasets.

    Reference: http://yifanhu.net/PUB/cf.pdf"""
    def __init__(self, opt_path, *args, **kwargs):
        Algo.__init__(self, *args, **kwargs)
        AlsOption.__init__(self, *args, **kwargs)
        Evaluable.__init__(self, *args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        Optimizable.__init__(self, *args, **kwargs)

        self.logger = log.get_logger('ALS')
        self.opt, self.opt_path = self.get_option(opt_path)
        self.obj = PyALS()
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
        self.logger.info('ALS(%s)' % json.dumps(self.opt, indent=2))
        if self.data:
            self.logger.info(self.data.show_info())
            assert self.data.data_type in ['matrix']

    def set_data(self, data):
        assert isinstance(data, aux.data.Data), 'Wrong instance: {}'.format(type(data))
        self.data = data

    def fast_similar(self, onoff=True):
        if onoff:
            self.FQ = self.normalize(self.Q)
        else:
            if hasattr(self, 'FQ'):
                del self.FQ

    def initialize(self):
        self.init_factors()

    def init_factors(self):
        assert self.data, 'Data is not setted'
        header = self.data.get_header()
        self.P = np.abs(np.random.normal(scale=1.0/(self.opt.d ** 2),
                                         size=(header['num_users'], self.opt.d)).astype("float32"), order='F')
        self.Q = np.abs(np.random.normal(scale=1.0/(self.opt.d ** 2),
                                         size=(header['num_items'], self.opt.d)).astype("float32"), order='F')
        self.obj.set_factors(self.P, self.Q)

    def _get_topk_recommendation(self, rows, topk):
        p = self.P[rows]
        topks = np.argsort(p.dot(self.Q.T), axis=1)[:, -topk:][:,::-1]
        return zip(rows, topks)

    def _get_most_similar(self, cols, topk):
        if hasattr(self, 'FQ'):
            q = self.FQ[cols]
            topks = np.argsort(q.dot(self.FQ.T), axis=1)[:, -topk:][:,::-1]
        else:
            q = self.Q[cols]
            dot = q.dot(self.Q.T)
            dot = dot / (norm(q) * norm(self.Q, axis=1))
            topks = np.argsort(dot, axis=1)[:, -topk:][:,::-1]
        return zip(cols, topks)

    def get_scores(self, row_col_pairs):
        rets = {(r, c): self.P[r].dot(self.Q[c]) for r, c in row_col_pairs}
        return rets

    def _get_buffer(self):
        buf = BufferedDataMatrix()
        buf.initialize(self.data)
        return buf

    def _iterate(self, buf, group='rowwise'):
        header = self.data.get_header()
        end = header['num_users'] if group == 'rowwise' else header['num_items']
        int_group = 0 if group == 'rowwise' else 1
        self.obj.precompute(int_group)
        err = 0.0
        update_t, feed_t, updated = 0, 0, 0
        buf.set_group(group)
        with log.pbar(log.DEBUG, desc='%s' % group,
                      total=header['num_nnz'], mininterval=30) as pbar:
            start_t = time.time()
            for sz in buf.fetch_batch():
                updated += sz
                feed_t += time.time() - start_t
                start_x, next_x, indptr, keys, vals = buf.get()
                start_t = time.time()
                err += self.obj.partial_update(start_x, next_x, indptr, keys, vals, int_group)
                update_t += time.time() - start_t
                pbar.update(sz)
            pbar.refresh()
        self.logger.debug('updated %s processed(%s) elapsed(data feed: %.3f update: %.3f)' % (group, updated, feed_t, update_t))
        return err

    def train(self):
        buf = self._get_buffer()
        best_loss, rmse, self.validation_result = 987654321.0, None, {}
        self.prepare_evaluation()
        self.initialize_tensorboard(self.opt.num_iters)
        for i in range(self.opt.num_iters):
            start_t = time.time()
            self._iterate(buf, group='rowwise')
            err = self._iterate(buf, group='colwise')
            rmse = (err / self.data.get_header()['num_nnz']) ** 0.5
            if self.opt.save_best and best_loss > rmse and (not self.opt.period or (i + 1) % self.opt.period == 0):
                best_loss = rmse
                self.save(self.model_path)
            self.logger.info('Iteration %d: RMSE %.3f Elapsed %.3f secs' % (i + 1, rmse, time.time() - start_t))
            metrics = {'rmse': rmse}
            if self.opt.validation:
                self.validation_result = self.get_validation_results()
                self.logger.info(self.validation_result)
                metrics.update({'val_%s' % k: v
                                for k, v in self.validation_result.items()})
            self.update_tensorboard_data(metrics)
        ret = {'rmse': rmse}
        ret.update({'val_%s' % k: v
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
        self.initializ()
        loss = self.train()
        loss['eval_time'] = time.time()
        loss['loss'] = loss.get(self.opt.optimize.loss)
        # TODO: deal with failture of training
        loss['status'] = HOPT_STATUS_OK
        self._optimize_loss = loss
        return loss

    def _get_data(self):
        return [('opt', self.opt),
                ('Q', self.Q),
                ('P', self.P)]

    def get_evaluation_metrics(self):
        return ['rmse', 'val_rmse', 'val_ndcg', 'val_map', 'val_accuracy', 'val_error']
