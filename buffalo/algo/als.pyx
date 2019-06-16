# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
# -*- coding: utf-8 -*-
from libcpp cimport bool
from libcpp.string cimport string
from eigency.core cimport MatrixXf, Map

import numpy as np
cimport numpy as np

import buffalo.data
from buffalo.misc import aux
from buffalo.algo.base import Algo
from buffalo.algo.options import AlsOption


cdef extern from "buffalo/algo_impl/als/als.hpp" namespace "als":
    cdef cppclass CALS:
        bool init(string) nogil except +
        void set_factors(Map[MatrixXf]&, Map[MatrixXf]&) nogil except +


cdef class PyALS:
    """CALS object holder"""
    cdef CALS* obj  # C-ALS object

    def __cinit__(self):
        self.obj = new CALS()

    def __dealloc__(self):
        del self.obj

    def init(self, option_path):
        return self.obj.init(option_path)

    def set_factors(self, np.ndarray P, np.ndarray Q):
        self.obj.set_factors(Map[MatrixXf](P), Map[MatrixXf](Q))


class AlsBuffer(object):
    def __init__(self, limit):
        self.resize(limit)
        self.logger = aux.get_logger('AlsBuffer')

    def resize(self, limit):
        self.index, self.limit = 0, limit
        self.U = np.zeros(shape=(limit,), dtype=np.int32)
        self.I = np.zeros(shape=(limit,), dtype=np.int32)
        self.V = np.zeros(shape=(limit,), dtype=np.float32)

    def add(self, key, keys, vals):
        if self.index == 0 and len(keys) >= self.limit:
            self.logger.warning('buffer size is too smaller, increase from {} to {}'.format(self.limit, len(keys) + 1))
            self.resize(len(keys) + 1)
        if self.index + len(keys) < self.limit:
            return False
        i = self.index
        sz = len(keys)
        self.U[i:i + sz] = [key] * sz
        self.I[i:i + sz] = keys
        self.V[i:i + sz] = vals
        self.index += sz
        return True

    def reset(self):
        self.U *= 0
        self.I *= 0
        self.V *= 0
        self.index = 0


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
        if kwargs.get('data_opt'):
            self.data = buffalo.data.load(kwargs['data_opt'])
            self.data.create()
        elif isinstance(kwargs.get('data'), buffalo.data.Data):
            self.data = kwargs.get('data')

    def set_data(self, data):
        assert isinstance(data, aux.data.Data), 'Wrong instance: {}'.format(type(data))
        self.data = data

    def init_factors(self):
        assert self.data, 'Data is not setted'
        header = self.data.get_header()
        self.P = np.abs(np.random.normal(scale=1.0/self.opt.d, size=(header['num_users'], self.opt.d)).astype("float32"))
        self.Q = np.abs(np.random.normal(scale=1.0/self.opt.d, size=(header['num_items'], self.opt.d)).astype("float32"))
        self.obj.set_factors(self.P, self.Q)

    def _get_buffer(self, est_chunk_size=None):
        if not est_chunk_size:
            est_chunk_size = max(int(((self.opt.batch_mb * 1024) / 12.)), 64)
        buf = AlsBuffer(est_chunk_size)
        return buf

    def _iterate(self, buf, axis='rowwise'):
        header = self.data.get_header()
        end = header['num_users'] if axis == 'rowwise' else header['num_items']
        for beg in range(end):
            key, keys, vals = self.data.get_data(beg, axis=axis)
            if not buf.add(key, keys, vals):
                # flush
                buf.reset()
                buf.add(key, keys, vals)
        if buf.index:
            # flush
            pass

    def train(self):
        buf = self._get_buffer()
        for i in range(self.opt.num_iters):
            self._iterate(buf, axis='rowwise')
            self._iterate(buf, axis='colwise')
