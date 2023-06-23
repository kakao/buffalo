# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: language=c++

cimport numpy as np
from libc.stdint cimport int32_t, int64_t
from libcpp.pair cimport pair
from libcpp.string cimport string

np.import_array()


cdef extern from "buffalo/algo_impl/als/als.hpp" namespace "als":
    cdef cppclass CALS:
        void release() nogil except +
        bint init(string) nogil except +
        void initialize_model(float*, int,
                              float*, int) nogil except +
        void precompute(int) nogil except +

        pair[double, double] partial_update(int,
                                            int,
                                            int64_t*,
                                            int32_t*,
                                            float*,
                                            int) nogil except +


cdef class CyALS:
    """CALS object holder"""
    cdef CALS* obj  # C-ALS object

    def __cinit__(self):
        self.obj = new CALS()

    def __dealloc__(self):
        self.obj.release()
        del self.obj

    def init(self, option_path):
        return self.obj.init(option_path)

    def initialize_model(self,
                         np.ndarray[np.float32_t, ndim=2] P,
                         np.ndarray[np.float32_t, ndim=2] Q):
        self.obj.initialize_model(&P[0, 0], P.shape[0],
                                  &Q[0, 0], Q.shape[0])

    def precompute(self, axis):
        self.obj.precompute(axis)

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
