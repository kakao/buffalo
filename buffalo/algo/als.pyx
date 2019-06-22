# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
# -*- coding: utf-8 -*-
import time
import json
import cython
from libcpp cimport bool
from libc.stdint cimport int64_t
from libcpp.string cimport string
from eigency.core cimport MatrixXf, Map, VectorXi, VectorXf

import numpy as np
cimport numpy as np

import buffalo.data
from buffalo.misc import aux
from buffalo.algo.base import Algo
from buffalo.data import BufferedDataMM
from buffalo.algo.options import AlsOption


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


class ALS(Algo, AlsOption):
    """Python implementation for C-ALS.

    Implementation of Collaborative Filtering for Implicit Feedback datasets.

    Reference: http://yifanhu.net/PUB/cf.pdf"""
    def __init__(self, opt_path, *args, **kwargs):
        super(ALS, self).__init__(*args, **kwargs)
        super(AlsOption, self).__init__(*args, **kwargs)
        self.logger = aux.get_logger('ALS')
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
        elif isinstance(data, buffalo.data.Data):
            self.data = data
        self.logger.info('ALS(%s)' % json.dumps(self.opt, indent=2))

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

    def _get_buffer(self, est_chunk_size=None):
        if not est_chunk_size:
            # 16 bytes(indptr8, keys4, vals4)
            est_chunk_size = max(int(((self.opt.batch_mb * 1024) / 16.)), 64)
        buf = BufferedDataMM()
        buf.initialize(est_chunk_size, self.data)
        return buf

    def _iterate(self, buf, axis='rowwise'):
        header = self.data.get_header()
        end = header['num_users'] if axis == 'rowwise' else header['num_items']
        int_axis = 0 if axis == 'rowwise' else 1
        self.obj.precompute(int_axis)
        err = 0.0
        start_t, update_t, time_feed_t = time.time(), 0, 0
        start_t = time.time()
        buf.set_axis(axis)
        for _ in buf.fetch_batch():
            time_feed_t += time.time() - start_t
            start_t = time.time()
            start_x, next_x, indptr, keys, vals = buf.get()
            err += self.obj.partial_update(start_x, next_x, indptr, keys, vals, int_axis)
            update_t += time.time() - start_t
        self.logger.debug('elapsed(data feed: %.3f update: %.3f)' % (time_feed_t, update_t))
        return err

    def train(self):
        buf = self._get_buffer()
        for i in range(self.opt.num_iters):
            start_t = time.time()
            self._iterate(buf, axis='rowwise')
            err = self._iterate(buf, axis='colwise')
            rmse = (err / self.data.get_header()['num_nnz']) ** 0.5
            self.logger.info('Iteration %d: RMSE %.3f Elapsed %.3f secs' % (i + 1, rmse, time.time() - start_t))
        self.obj.release()
