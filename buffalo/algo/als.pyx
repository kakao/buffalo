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

import buffalo.data
from buffalo.data.base import Data
from buffalo.misc import aux, log
from buffalo.algo.base import Algo
from buffalo.evaluate import Evaluable
from buffalo.algo.options import AlsOption
from buffalo.data.buffered_data import BufferedDataMM


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
        del self.obj

    def release(self):
        self.obj.release()

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


class ALS(Algo, AlsOption, Evaluable):
    """Python implementation for C-ALS.

    Implementation of Collaborative Filtering for Implicit Feedback datasets.

    Reference: http://yifanhu.net/PUB/cf.pdf"""
    def __init__(self, opt_path, *args, **kwargs):
        super(ALS, self).__init__(*args, **kwargs)
        super(AlsOption, self).__init__(*args, **kwargs)
        self.logger = log.get_logger('ALS')
        self.opt, self.opt_path = self.get_option(opt_path)
        self.obj = PyALS()
        assert self.obj.init(bytes(self.opt_path, 'utf-8')),\
            'cannot parse option file: %s' % opt_path
        self.data = None
        data = kwargs.get('data')
        data_opt = kwargs.get('data_opt')
        if data_opt:
            self.data = buffalo.data.load(data_opt)
            self.data.create()
        elif isinstance(data, Data):
            self.data = data
        self.logger.info('ALS(%s)' % json.dumps(self.opt, indent=2))
        self.logger.info(self.data.show_info())

    def set_data(self, data):
        assert isinstance(data, aux.data.Data), 'Wrong instance: {}'.format(type(data))
        self.data = data

    def init_factors(self):
        assert self.data, 'Data is not setted'
        header = self.data.get_header()
        self.P = np.abs(np.random.normal(scale=1.0/(self.opt.d ** 2),
                                         size=(header['num_users'], self.opt.d)).astype("float32"), order='F')
        self.Q = np.abs(np.random.normal(scale=1.0/(self.opt.d ** 2),
                                         size=(header['num_items'], self.opt.d)).astype("float32"), order='F')
        self.obj.set_factors(self.P, self.Q)

    def get_topk_recommendation(self, rows, topk):
        p = self.P[rows]
        topks = np.argsort(p.dot(self.Q.T), axis=1)[:, -topk:][:,::-1]
        return topks

    def get_scores(self, row_col_pairs):
        rets = {(r, c): self.P[r].dot(self.Q[c]) for r, c in row_col_pairs}
        return rets

    def _get_buffer(self):
        buf = BufferedDataMM()
        buf.initialize(self.data)
        return buf

    def _iterate(self, buf, axis='rowwise'):
        header = self.data.get_header()
        end = header['num_users'] if axis == 'rowwise' else header['num_items']
        int_axis = 0 if axis == 'rowwise' else 1
        self.obj.precompute(int_axis)
        err = 0.0
        update_t, feed_t, updated = 0, 0, 0
        buf.set_axis(axis)
        with log.pbar(log.DEBUG, desc='%s' % axis,
                      total=header['num_nnz'], mininterval=30) as pbar:
            start_t = time.time()
            for sz in buf.fetch_batch():
                updated += sz
                feed_t += time.time() - start_t
                start_x, next_x, indptr, keys, vals = buf.get()
                start_t = time.time()
                err += self.obj.partial_update(start_x, next_x, indptr, keys, vals, int_axis)
                update_t += time.time() - start_t
                pbar.update(sz)
            pbar.refresh()
        self.logger.debug('updated %s processed(%s) elapsed(data feed: %.3f update: %.3f)' % (axis, updated, feed_t, update_t))
        return err

    def train(self):
        buf = self._get_buffer()
        for i in range(self.opt.num_iters):
            start_t = time.time()
            self._iterate(buf, axis='rowwise')
            err = self._iterate(buf, axis='colwise')
            rmse = (err / self.data.get_header()['num_nnz']) ** 0.5
            self.logger.info('Iteration %d: RMSE %.3f Elapsed %.3f secs' % (i + 1, rmse, time.time() - start_t))
            if self.opt.validation:
                self.logger.info(self.show_validation_results())
        self.obj.release()
