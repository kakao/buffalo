# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: language=c++

cimport numpy as np
from libc.stdint cimport int32_t, int64_t
from libcpp.pair cimport pair
from libcpp.string cimport string

np.import_array()


cdef extern from "buffalo/algo_impl/eals/eals.hpp" namespace "eals":
    cdef cppclass CEALS:
        bint init(string) nogil except +
        void initialize_model(float*, float*, float*, int, int) nogil except +
        void precompute_cache(int, int64_t*, int32_t*, int) nogil except +
        bint update(int64_t*, int32_t*, float*, int) nogil except +
        pair[float, float] estimate_loss(int,
                                         int64_t*,
                                         int32_t*,
                                         float*,
                                         int) nogil except +

cdef class CyEALS:
    """CEALS object holder"""
    cdef CEALS* _obj  # C-EALS object

    def __cinit__(self):
        self._obj = new CEALS()

    def __dealloc__(self):
        del self._obj

    def init(self, option_path):
        return self._obj.init(option_path)

    def initialize_model(self,
                         np.ndarray[np.float32_t, ndim=2] P,
                         np.ndarray[np.float32_t, ndim=2] Q,
                         np.ndarray[np.float32_t, ndim=1] C):
        self._obj.initialize_model(&P[0, 0], &Q[0, 0], &C[0], P.shape[0], Q.shape[0])

    def precompute_cache(self,
                         int nnz,
                         np.ndarray[np.int64_t, ndim=1] indptr,
                         np.ndarray[np.int32_t, ndim=1] keys,
                         int axis):
        self._obj.precompute_cache(nnz, &indptr[0], &keys[0], axis)

    def update(self,
               np.ndarray[np.int64_t, ndim=1] indptr,
               np.ndarray[np.int32_t, ndim=1] keys,
               np.ndarray[np.float32_t, ndim=1] vals,
               int axis):
        return self._obj.update(&indptr[0], &keys[0], &vals[0], axis)

    def estimate_loss(self,
                      int nnz,
                      np.ndarray[np.int64_t, ndim=1] indptr,
                      np.ndarray[np.int32_t, ndim=1] keys,
                      np.ndarray[np.float32_t, ndim=1] vals,
                      int axis):
        return self._obj.estimate_loss(nnz, &indptr[0], &keys[0], &vals[0], axis)
