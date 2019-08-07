# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
# -*- coding: utf-8 -*-
import time
import json
import logging

import tqdm
import cython
from libc.stdint cimport int64_t
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as bool_t

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


cdef extern from "buffalo/algo_impl/cfr.hpp" namespace "cfr":
    cdef cppclass CCFR:
        CCFR(int dim, int num_threads, int num_cg_max_iters,
             float alpha, float l, float cg_tolerance,
             float reg_u, float reg_i, float reg_c,
             bool_t compute_loss, string optimizer);
        void set_embedding(float*, int, string) nogil except +
        void precompute(string) nogil except +
        double partial_update_user(int, int,
                                   int64_t*, int*, float*) nogil except +
        double partial_update_item(int, int,
                                   int64_t*, int*, float*,
                                   int64_t*, int*, float*) nogil except +
        double partial_update_context(int, int,
                                      int64_t*, int*, float*) nogil except +


cdef class CyCFR:
    """CCFR object holder"""
    cdef CCFR* obj  # C-ALS object

    def __cinit__(self, dim, num_threads, num_cg_max_iters,
                  alpha, l, cg_tolerance, reg_u, reg_i, reg_c,
                  compute_loss, optimizer):
        self.obj = new CCFR(dim ,num_threads, num_cg_max_iters,
                            alpha, l, cg_tolerance, reg_u, reg_i, reg_c,
                            compute_loss, optimizer)

    def __dealloc__(self):
        del self.obj

    def precompute(self, obj_type):
        self.obj.precompute(obj_type)

    def set_embedding(self,
                      np.ndarray[np.float32_t, ndim=2] F, obj_type):
        self.obj.set_factors(&F[0, 0], F.shape[0], obj_type)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def partial_update_user(self, int start_x, int next_x,
                            np.ndarray[np.int64_t, ndim=1] indptrs,
                            np.ndarray[np.int32_t, ndim=1] keys,
                            np.ndarray[np.float32_t, ndim=1] vals):
        return self.obj.partial_update_user(start_x, next_x,
                                            &indptrs[0], &keys[0], &vals[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def partial_update_item(self, int start_x, int next_x,
                            np.ndarray[np.int64_t, ndim=1] indptrs_u,
                            np.ndarray[np.int32_t, ndim=1] keys_u,
                            np.ndarray[np.float32_t, ndim=1] vals_u,
                            np.ndarray[np.int64_t, ndim=1] indptrs_c,
                            np.ndarray[np.int32_t, ndim=1] keys_c,
                            np.ndarray[np.float32_t, ndim=1] vals_c):
        return self.obj.partial_update_item(start_x, next_x,
                                            &indptrs_u[0], &keys_u[0], &vals_u[0],
                                            &indptrs_c[0], &keys_c[0], &vals_c[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def partial_update_context(self, int start_x, int next_x,
                               np.ndarray[np.int64_t, ndim=1] indptrs,
                               np.ndarray[np.int32_t, ndim=1] keys,
                               np.ndarray[np.float32_t, ndim=1] vals):
        return self.obj.partial_update_context(start_x, next_x,
                                               &indptrs[0], &keys[0], &vals[0])
