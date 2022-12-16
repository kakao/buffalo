# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
from libc.stdint cimport int64_t
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "fileio.hpp" namespace "fileio":
    vector[string] _chunking_into_bins(string, string, int64_t, int, int, int)
    int64_t _parallel_build_sppmi(string, string, int64_t, int, int, int)
    vector[string] _sort_and_compressed_binarization(string, string, int64_t, int, int, int)


def chunking_into_bins(path, to_dir, total_lines, num_chunks, sep_idx, num_workers):
    assert num_chunks % num_workers == 0, 'The number of chunks must be a multiple of the number of workers'
    return _chunking_into_bins(bytes(path, 'utf-8'),
                               bytes(to_dir, 'utf-8'),
                               total_lines,
                               num_chunks,
                               sep_idx,
                               num_workers)


def parallel_build_sppmi(from_path, to_path, total_lines, num_items, k, num_workers):
    return _parallel_build_sppmi(bytes(from_path, 'utf-8'),
                                 bytes(to_path, 'utf-8'),
                                 total_lines,
                                 num_items,
                                 k,
                                 num_workers)


def sort_and_compressed_binarization(path, to_dir, total_lines, max_key, sort_key, num_workers):
    return _sort_and_compressed_binarization(
        bytes(path, 'utf-8'),
        bytes(to_dir, 'utf-8'),
        total_lines,
        max_key,
        sort_key,
        num_workers)
