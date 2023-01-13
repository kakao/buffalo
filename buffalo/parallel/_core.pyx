# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
import cython

from libc.stdint cimport int32_t, int64_t
from libcpp cimport bool as bool_t
from libcpp.string cimport string
from libcpp.vector cimport vector

import numpy as np

cimport numpy as np


cdef extern from "_core.hpp":
    cdef void _quickselect "parallel::quickselect"(float*, int, int, int32_t*, int, bool_t, int) nogil except +

    cdef void _dot_topn "parallel::dot_topn"(int32_t*, int,
                                             float*, int, int,
                                             float*, int, int,
                                             float*, int,
                                             int32_t*, float*,
                                             int32_t*, int,
                                             int, int) nogil except +


@cython.boundscheck(False)
@cython.wraparound(False)
def quickselect(np.ndarray[np.float32_t, ndim=2] scores,
                np.ndarray[np.int32_t, ndim=2] result,
                sorted, num_threads):
    _quickselect(&scores[0, 0], scores.shape[0], scores.shape[1],
                 &result[0, 0], result.shape[1], sorted, num_threads)


@cython.boundscheck(False)
@cython.wraparound(False)
def dot_topn(np.ndarray[np.int32_t, ndim=1] indexes,
             np.ndarray[np.float32_t, ndim=2] P,
             np.ndarray[np.float32_t, ndim=2] Q,
             np.ndarray[np.float32_t, ndim=2] Qb,
             np.ndarray[np.int32_t, ndim=2]  out_keys,
             np.ndarray[np.float32_t, ndim=2]  out_scores,
             np.ndarray[np.int32_t, ndim=1] pool,
             k, num_threads):
    qb_shape = Qb.shape[0] if Qb.shape[1] != 0 else 0
    _dot_topn(&indexes[0], indexes.shape[0],
              &P[0, 0], P.shape[0], P.shape[1],
              &Q[0, 0], Q.shape[0], Q.shape[1],
              &Qb[0, 0], qb_shape,
              &out_keys[0, 0], &out_scores[0, 0],
              &pool[0], pool.shape[0],
              k, num_threads)
