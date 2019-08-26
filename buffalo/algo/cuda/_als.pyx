# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
# -*- coding: utf-8 -*-
import cython
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.string cimport string
from libc.stdint cimport int32_t

import numpy as np
cimport numpy as np


cdef extern from "buffalo/cuda/als/als.hpp" namespace "cuda_als":
    cdef cppclass CuALS:
        CuALS() nogil except +
        bool init(string) nogil except +
        void initialize_model(float*, int,
                              float*, int) nogil except +
        void precompute(int) nogil except +
        void synchronize(int, bool) nogil except +
        pair[double, double] partial_update(int, int,
                             int32_t*, int32_t*,
                             float*, int) nogil except +
        int get_vdim() nogil except +


cdef class CyALS:
    """CuALS object holder"""
    cdef CuALS* obj  # Cuda-ALS object

    def __cinit__(self):
        self.obj = new CuALS()

    def __dealloc__(self):
        del self.obj

    def init(self, opt_path):
        return self.obj.init(opt_path)

    def initialize_model(self,
                         np.ndarray[np.float32_t, ndim=2] P,
                         np.ndarray[np.float32_t, ndim=2] Q):
        self.obj.initialize_model(&P[0, 0], P.shape[0],
                                  &Q[0, 0], Q.shape[0])

    def precompute(self, axis):
        self.obj.precompute(axis)

    def synchronize(self, axis, device_to_host):
        self.obj.synchronize(axis, device_to_host)

    def get_vdim(self):
        return self.obj.get_vdim()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def partial_update(self,
                       int start_x,
                       int next_x,
                       np.ndarray[np.int32_t, ndim=1] indptr,
                       np.ndarray[np.int32_t, ndim=1] keys,
                       np.ndarray[np.float32_t, ndim=1] vals,
                       int axis):
        return self.obj.partial_update(start_x,
                                       next_x,
                                       &indptr[0],
                                       &keys[0],
                                       &vals[0],
                                       axis)
