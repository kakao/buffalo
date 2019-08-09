# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
#-*- coding: utf-8 -*-
from libcpp.vector cimport vector
from libcpp.string cimport string


cdef extern from "fileio.hpp" namespace "fileio":
    vector[string] _chunking_file(string, string, long, int, int, int)


def chunking_file(path, to_dir, total_lines, num_chunks, sep_idx, num_workers):
    assert num_chunks % num_workers == 0, 'The number of chunks must be a multiple of the number of workers'
    return _chunking_file(bytes(path, 'utf-8'),
                          bytes(to_dir, 'utf-8'),
                          total_lines,
                          num_chunks,
                          sep_idx,
                          num_workers)
