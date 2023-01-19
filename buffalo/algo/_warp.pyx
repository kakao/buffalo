# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
import cython

from libc.stdint cimport int32_t, int64_t
from libcpp cimport bool
from libcpp.string cimport string

import numpy as np

cimport numpy as np


cdef extern from "buffalo/algo_impl/warp/warp.hpp" namespace "warp":
    cdef cppclass CWARP:
        void release() nogil except +
        bool init(string) nogil except +
        void initialize_model(float*, int32_t,
                              float*, int32_t,
                              float*,
                              int64_t) nogil except +
        void set_cumulative_table(int64_t*, int)
        void launch_workers()
        void add_jobs(int,
                      int,
                      int64_t*,
                      int32_t*) nogil except +
        double compute_loss(int32_t,
                            int32_t*,
                            int32_t*,
                            int32_t*) nogil except +
        double join()
        void wait_until_done()
        void update_parameters()


cdef class CyWARP:
    """CWARP object holder"""
    cdef CWARP* obj  # C-BPRMF object

    def __cinit__(self):
        self.obj = new CWARP()

    def __dealloc__(self):
        self.obj.release()
        del self.obj

    def init(self, option_path):
        return self.obj.init(option_path)

    def initialize_model(self,
                         np.ndarray[np.float32_t, ndim=2] P,
                         np.ndarray[np.float32_t, ndim=2] Q,
                         np.ndarray[np.float32_t, ndim=2] Qb,
                         int64_t num_total_samples):
        self.obj.initialize_model(&P[0, 0], P.shape[0],
                                  &Q[0, 0], Q.shape[0],
                                  &Qb[0, 0],
                                  num_total_samples)

    def set_cumulative_table(self,
                             np.ndarray[np.int64_t, ndim=1] cum_table,
                             int cum_table_size):
        self.obj.set_cumulative_table(&cum_table[0], cum_table_size)

    def launch_workers(self):
        self.obj.launch_workers()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_jobs(self,
                 int start_x,
                 int next_x,
                 np.ndarray[np.int64_t, ndim=1] indptr,
                 np.ndarray[np.int32_t, ndim=1] positives):
        self.obj.add_jobs(start_x,
                          next_x,
                          &indptr[0],
                          &positives[0])

    def compute_loss(self,
                     np.ndarray[np.int32_t, ndim=1] users,
                     np.ndarray[np.int32_t, ndim=1] positives,
                     np.ndarray[np.int32_t, ndim=1] negatives):
        return self.obj.compute_loss(users.shape[0],
                                     &users[0],
                                     &positives[0],
                                     &negatives[0])

    def join(self):
        return self.obj.join()

    def wait_until_done(self):
        self.obj.wait_until_done()

    def update_parameters(self):
        self.obj.update_parameters()
