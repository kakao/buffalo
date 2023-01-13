# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
import cython

from libc.stdint cimport int32_t, int64_t
from libcpp cimport bool as bool_t
from libcpp.string cimport string
from libcpp.vector cimport vector

import numpy as np

cimport numpy as np


cdef extern from "buffalo/algo_impl/cfr/cfr.hpp" namespace "cfr":
    cdef cppclass CCFR:
        bool_t init(string) nogil except +
        void set_embedding(float*, int, string) nogil except +
        void precompute(string) nogil except +
        double partial_update_user(int, int,
                                   int64_t*, int32_t*, float*) nogil except +
        double partial_update_item(int, int,
                                   int64_t*, int32_t*, float*,
                                   int64_t*, int32_t*, float*) nogil except +
        double partial_update_context(int, int,
                                      int64_t*, int32_t*, float*) nogil except +


cdef class CyCFR:
    """CCFR object holder"""
    cdef CCFR* obj  # C-CFR object

    def __cinit__(self):
        self.obj = new CCFR()

    def __dealloc__(self):
        del self.obj

    def init(self, opt_path):
        return self.obj.init(opt_path)

    def precompute(self, obj_type):
        self.obj.precompute(obj_type)

    def set_embedding(self,
                      np.ndarray[np.float32_t, ndim=2] F, obj_type):
        self.obj.set_embedding(&F[0, 0], F.shape[0], obj_type)

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
