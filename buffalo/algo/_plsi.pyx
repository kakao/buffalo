# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
import cython
import numpy as np

cimport numpy as np
from libc.stdint cimport int32_t, int64_t
from libcpp cimport bool as bool_t
from libcpp.string cimport string


cdef extern from "buffalo/algo_impl/plsi/plsi.hpp" namespace "plsi":
    cdef cppclass CPLSI:
        bool_t init(string) nogil except +
        void release() nogil except +
        void swap() nogil except +
        void reset() nogil except +
        void initialize_model(float*, int, float*, int) nogil except +
        float partial_update(int, int, int64_t*, int32_t*, float*) nogil except +
        void normalize(float, float) nogil except +


cdef class CyPLSI:
    """CPLSI object holder"""
    cdef CPLSI* obj  # C-PLSI object

    def __cinit__(self):
        self.obj = new CPLSI()

    def __dealloc__(self):
        self.obj.release()
        del self.obj

    def init(self, opt_path):
        return self.obj.init(opt_path)

    def swap(self):
        self.obj.swap()

    def release(self):
        self.obj.release()

    def reset(self):
        self.obj.reset()

    def initialize_model(self, np.ndarray[np.float32_t, ndim=2] P,
                         np.ndarray[np.float32_t, ndim=2] Q):
        self.obj.initialize_model(&P[0, 0], P.shape[0],
                                  &Q[0, 0], Q.shape[0])

    def normalize(self, alpha1, alpha2):
        self.obj.normalize(alpha1, alpha2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def partial_update(self, int start_x, int next_x,
                       np.ndarray[np.int64_t, ndim=1] indptr,
                       np.ndarray[np.int32_t, ndim=1] keys,
                       np.ndarray[np.float32_t, ndim=1] vals):
        return self.obj.partial_update(start_x, next_x, &indptr[0], &keys[0], &vals[0])
