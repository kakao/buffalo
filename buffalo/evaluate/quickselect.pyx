# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
#-*- coding: utf-8 -*-
import cython
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as bool_t
from libc.stdint cimport int32_t, int64_t

import numpy as np
cimport numpy as np


cdef extern from "quickselect.hpp" namespace "evaluate":
    void quickselect(float*, int, int, int*, int, bool_t, int)


@cython.boundscheck(False)
@cython.wraparound(False)
def quickselect(np.ndarray[np.float32_t, ndim=2] scores,
                np.ndarray[np.int32_t, ndim=2] result,
                bool_t sorted, int num_threads):
    quickselect(&scores[0, 0], scores.shape[0], scores.shape[1],
                &result[0, 0], result.shape[1], sorted, num_threads)
