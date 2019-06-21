# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
# -*- coding: utf-8 -*-
import time
from libcpp cimport bool
from libcpp.string cimport string
from eigency.core cimport MatrixXf, Map, VectorXi, VectorXf

import numpy as np
cimport numpy as np

import buffalo.data
from buffalo.misc import aux
from buffalo.algo.base import Algo
from buffalo.algo.options import AlsOption


cdef extern from "buffalo/algo_impl/als/als.hpp" namespace "als":
    cdef cppclass CALS:
        void release() nogil except +
        bool init(string) nogil except +
        void set_factors(Map[MatrixXf]&,
                         Map[MatrixXf]&) nogil except +
        void precompute(int) nogil except +
        double partial_update(Map[VectorXi]&, Map[VectorXi]&,
                              Map[VectorXi]&, Map[VectorXf]&, int) nogil except +


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

    def partial_update(self,
                       np.ndarray[np.int32_t, ndim=1] indptr,
                       np.ndarray[np.int32_t, ndim=1] rows,
                       np.ndarray[np.int32_t, ndim=1] keys,
                       np.ndarray[np.float32_t, ndim=1] vals,
                       int axis):
        return self.obj.partial_update(Map[VectorXi](indptr),
                                       Map[VectorXi](rows),
                                       Map[VectorXi](keys),
                                       Map[VectorXf](vals),
                                       axis)


class DataBuffer(object):
    def __init__(self, limit):
        self.resize(limit)
        self.logger = aux.get_logger('DataBuffer')

    def resize(self, limit):
        self.index, self.limit = 0, limit
        self.rows = []
        self.indptr = []
        self.keys = np.zeros(shape=(limit,), dtype=np.int32, order='F')
        self.vals = np.zeros(shape=(limit,), dtype=np.float32, order='F')

    def add(self, key, keys, vals):
        if self.index == 0 and len(keys) >= self.limit:
            self.logger.warning('buffer size is too smaller, increase from {} to {}'.format(self.limit, len(keys) + 1))
            self.resize(len(keys) + 1)
        if self.index + len(keys) >= self.limit:
            return False
        i = self.index
        sz = len(keys)
        self.rows.append(key)
        self.indptr.append(i + sz)
        self.keys[i:i + sz] = keys
        self.vals[i:i + sz] = vals
        self.index += sz
        return True

    def reset(self):
        self.rows = []
        self.indptr = []
        self.keys *= 0
        self.vals *= 0
        self.index = 0

    def get(self):
        # TODO: Checkout long long structure for Eigen/Eigency
        return (np.array(self.indptr, dtype=np.int32),
                np.array(self.rows, dtype=np.int32),
                self.keys,
                self.vals)


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
            est_chunk_size = max(int(((self.opt.batch_mb * 1024) / 12.)), 64)
        buf = DataBuffer(est_chunk_size)
        return buf

    def _iterate(self, buf, axis='rowwise'):
        header = self.data.get_header()
        end = header['num_users'] if axis == 'rowwise' else header['num_items']
        int_axis = 0 if axis == 'rowwise' else 1
        self.obj.precompute(int_axis)
        err = 0.0
        for beg in range(end):
            key, keys, vals = self.data.get_data(beg, axis=axis)
            if not buf.add(key, keys, vals):
                indptr, R, K, V = buf.get()
                err += self.obj.partial_update(indptr, R, K, V, int_axis)
                buf.reset()
                buf.add(key, keys, vals)
        if buf.index:
            indptr, R, K, V = buf.get()
            err += self.obj.partial_update(indptr, R, K, V, int_axis)
            buf.reset()
        return err

    def train(self):
        buf = self._get_buffer()
        for i in range(self.opt.num_iters):
            start_t = time.time()
            err = self._iterate(buf, axis='rowwise')
            err = self._iterate(buf, axis='colwise')
            self.logger.info('Iteration %d: Error %.3f Elapsed %.3f secs' % (i + 1, err, time.time() - start_t))
        self.obj.release()
