# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
import cython

from libc.stdint cimport int32_t, int64_t
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.string cimport string

import numpy as np

cimport numpy as np


cdef extern from "buffalo/cuda/bpr/bpr.hpp" namespace "cuda_bpr":
    cdef cppclass CuBPR:
        CuBPR() nogil except +
        bool init(string) nogil except +
        void set_placeholder(int64_t* indptr, size_t batch_size) nogil except +
        void set_cumulative_table(int64_t*) nogil except +
        void initialize_model(float*, int,
                              float*, float*, int, int64_t, bool) nogil except +
        pair[double, double] partial_update(int, int,
                                            int64_t*, int32_t*) nogil except +
        double compute_loss(int, int32_t*, int32_t*, int32_t*) nogil except +
        int get_vdim() nogil except +
        void synchronize(bool)


cdef class CyBPR:
    """CuBPR object holder"""
    cdef CuBPR* obj  # Cuda-ALS object

    def __cinit__(self):
        self.obj = new CuBPR()

    def __dealloc__(self):
        del self.obj

    def init(self, opt_path):
        return self.obj.init(opt_path)

    def initialize_model(self,
                         np.ndarray[np.float32_t, ndim=2] P,
                         np.ndarray[np.float32_t, ndim=2] Q,
                         np.ndarray[np.float32_t, ndim=2] Qb,
                         num_nnz, set_gpu=False):
        self.obj.initialize_model(&P[0, 0], P.shape[0],
                                  &Q[0, 0], &Qb[0, 0], Q.shape[0], num_nnz, set_gpu)

    def set_placeholder(self, np.ndarray[np.int64_t, ndim=1] indptr, batch_size):
        self.obj.set_placeholder(&indptr[0], batch_size)

    def set_cumulative_table(self, np.ndarray[np.int64_t, ndim=1] sampling_table, size):
        self.obj.set_cumulative_table(&sampling_table[0])

    def get_vdim(self):
        return self.obj.get_vdim()

    def synchronize(self, device_to_host):
        self.obj.synchronize(device_to_host)

    def update_parameters(self):
        self.synchronize(True)

    def wait_until_done(self):
        return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_jobs(self,
                 int start_x,
                 int next_x,
                 np.ndarray[np.int64_t, ndim=1] indptr,
                 np.ndarray[np.int32_t, ndim=1] keys):
        return self.obj.partial_update(start_x,
                                       next_x,
                                       &indptr[0],
                                       &keys[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_loss(self,
                     np.ndarray[np.int32_t, ndim=1] user,
                     np.ndarray[np.int32_t, ndim=1] pos,
                     np.ndarray[np.int32_t, ndim=1] neg):
        return self.obj.compute_loss(user.shape[0], &user[0], &pos[0], &neg[0])
