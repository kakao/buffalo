# cython: experimental_cpp_class_def=True, language_level=2
# distutils: language=c++
# -*- coding: utf-8 -*-


cdef extern from "buffalo/misc/log.hpp":
    cdef cppclass _BuffaloLogger "BuffaloLogger":
        void set_log_level(int)

cdef class PyBuffaloLog:
    """CALS object holder"""
    cdef _BuffaloLogger* obj

    def __cinit__(self):
        self.obj = new _BuffaloLogger()

    def __dealloc__(self):
        del self.obj

    def set_log_level(self, lvl):
        self.obj.set_log_level(lvl)


def set_log_level(lvl):
    PyBuffaloLog().set_log_level(lvl)
