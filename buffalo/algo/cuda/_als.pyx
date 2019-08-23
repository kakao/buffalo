# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
# -*- coding: utf-8 -*-
import cython
from libcpp cimport bool
from libcpp.string cimport string

import numpy as np
cimport numpy as np


cdef extern from "buffalo/cuda/als/als.h" namespace "als":
    cdef cppclass _CuALS "CuALS":
        void set_options(bool, int, int,
                         float, float, float, float, float) nogil except +
        void initialize_model(float*, int,
                              float*, int) nogil except +
        void precompute(int) nogil except +
        float partial_update(int, int,
                             int*, int*,
                             float*, int) nogil except +
        int get_vdim() nogil except +


cdef class CuALS:
    """CuALS object holder"""
    cdef _CuALS* obj  # Cuda-ALS object

    def __cinit__(self):
        self.obj = new _CuALS()

    def __dealloc__(self):
        del self.obj

    def set_options(self, compute_loss, dim, num_cg_max_iters,
                    alpha, reg_u, reg_i, cg_tolerance, eps):
        self.obj.set_options(compute_loss, dim, alpha, reg_u, reg_i,
                             cg_tolerance, eps)

    def initialize_model(self,
                         np.ndarray[np.float32, ndim=2] P,
                         np.ndarray[np.float32, ndim=2] Q):
        self.obj.initialize_model(&P[0, 0], P.shape[0],
                                  &Q[0, 0], Q.shape[0])

    def precompute(self, axis):
        self.obj.precompute(axis)

    def get_vdim(self):
        return self.obj.get_vdim()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def partial_update(self,
                       int start_x,
                       int next_x,
                       np.ndarray[np.int32, ndim=1] indptr,
                       np.ndarray[np.int32, ndim=1] keys,
                       np.ndarray[np.float32, ndim=1] vals,
                       int axis):
        return self.obj.partial_update(start_x,
                                       next_x,
                                       &indptr[0],
                                       &keys[0],
                                       &vals[0],
                                       axis)
