# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
import cython

from libc.stdint cimport int32_t, int64_t, uint32_t
from libcpp cimport bool
from libcpp.string cimport string

import numpy as np

cimport numpy as np


cdef extern from "buffalo/algo_impl/w2v/w2v.hpp" namespace "w2v":
    cdef cppclass CW2V:
        void release() nogil except +
        bool init(string) nogil except +
        void initialize_model(float*, int32_t,
                              int32_t*,
                              uint32_t*,
                              int32_t*,
                              int64_t) nogil except +
        void launch_workers()
        void add_jobs(int,
                      int,
                      int64_t*,
                      int32_t*) nogil except +
        double join()


cdef class CyW2V:
    """CW2V object holder"""
    cdef CW2V* obj  # C-W2V object

    def __cinit__(self):
        self.obj = new CW2V()

    def __dealloc__(self):
        self.obj.release()
        del self.obj

    def init(self, option_path):
        return self.obj.init(option_path)

    def initialize_model(self,
                         np.ndarray[np.float32_t, ndim=2] L0,
                         np.ndarray[np.int32_t, ndim=1] index,
                         np.ndarray[np.uint32_t, ndim=1] scale,
                         np.ndarray[np.int32_t, ndim=1] dist,
                         int64_t total_word_count):
        self.obj.initialize_model(&L0[0, 0],
                                  L0.shape[0],
                                  &index[0],
                                  &scale[0],
                                  &dist[0],
                                  total_word_count)

    def launch_workers(self):
        self.obj.launch_workers()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_jobs(self,
                 int start_x,
                 int next_x,
                 np.ndarray[np.int64_t, ndim=1] indptr,
                 np.ndarray[np.int32_t, ndim=1] sequences):
        self.obj.add_jobs(start_x,
                          next_x,
                          &indptr[0],
                          &sequences[0])

    def join(self):
        return self.obj.join()
