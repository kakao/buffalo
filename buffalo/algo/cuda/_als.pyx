# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
import cython

from libc.stdint cimport int32_t, int64_t
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.string cimport string

import numpy as np

cimport numpy as np


cdef extern from "buffalo/cuda/als/als.hpp" namespace "cuda_als":
    cdef cppclass CuALS:
        CuALS() nogil except +
        bool init(string) nogil except +
        void set_placeholder(int64_t* lindptr, int64_t* rindptr, size_t batch_size)
        void initialize_model(float*, int,
                              float*, int) nogil except +
        void precompute(int) nogil except +
        pair[double, double] partial_update(int, int,
                                            int64_t*, int32_t*, float*, int) nogil except +
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

    def set_placeholder(self, np.ndarray[np.int64_t, ndim=1] lindptr,
                        np.ndarray[np.int64_t, ndim=1] rindptr,
                        batch_size):
        self.obj.set_placeholder(&lindptr[0], &rindptr[0], batch_size)

    def precompute(self, axis):
        self.obj.precompute(axis)

    def get_vdim(self):
        return self.obj.get_vdim()

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
                                       &keys[0],
                                       &vals[0],
                                       axis)
