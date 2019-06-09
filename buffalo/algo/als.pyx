# cython: experimental_cpp_class_def=True, language_level=2
# distutils: language=c++
# -*- coding: utf-8 -*-
from libcpp cimport bool
from libcpp.string cimport string
from eigency.core cimport MatrixXf, Map

import numpy as np
cimport numpy as np

from buffalo.misc import aux


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


class ALS:
    """Python implementation for C-ALS.

    Implementation of Collaborative Filtering for Implicit Feedback datasets.

    Reference: http://yifanhu.net/PUB/cf.pdf"""
    def __init__(self, opt_path, **kwargs):
        self.logger = aux.get_logger('ALS')
        self.opt = aux.Option(opt_path)
        self.obj = PyALS()
        assert self.obj.init(opt_path), 'cannot parse option file: %s' % opt_path

    def init_factors(self):
        self.P = np.abs(np.random.normal(scale=1.0/self.opt.d, size=(200000, self.opt.d)).astype("float32"))
        self.Q = np.abs(np.random.normal(scale=1.0/self.opt.d, size=(20000, self.opt.d)).astype("float32"))
        self.obj.set_factors(self.P, self.Q)
