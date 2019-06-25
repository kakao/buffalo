# cython: experimental_cpp_class_def=True
# distutils: language=c++
# -*- coding: utf-8 -*-
import io
import os
import sys
from contextlib import contextmanager

import tqdm

NOTSET = 0
WARN = 1
INFO = 2
DEBUG = 3
TRACE = 4


cdef extern from "buffalo/misc/log.hpp":
    cdef cppclass _BuffaloLogger "BuffaloLogger":
        void set_log_level(int)
        int get_log_level()


cdef class PyBuffaloLog:
    """CALS object holder"""
    cdef _BuffaloLogger* obj

    def __cinit__(self):
        self.obj = new _BuffaloLogger()

    def __dealloc__(self):
        del self.obj

    def set_log_level(self, lvl):
        self.obj.set_log_level(lvl)

    def get_log_level(self):
        return self.obj.get_log_level()


def set_log_level(lvl):
    PyBuffaloLog().set_log_level(lvl)


def get_log_level():
    return PyBuffaloLog().get_log_level()


class TqdmLogger(io.StringIO):
    logger = None
    buf = ''

    def __init__(self, logger):
        super(TqdmLogger, self).__init__()
        self.logger = logger

    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        if self.buf.strip():
            self.logger(self.buf)


@contextmanager
def pbar(logger, **kwargs):
    if 'total' not in kwargs:
        fmt = '{desc}: {n_fmt}/{total_fmt} {elapsed}<{remaining}'
    else:
        fmt = '{l_bar} {elapsed}<{remaining}'
    if 'bar_format' not in kwargs:
        kwargs['bar_format'] = fmt
    if 'desc' not in kwargs:
        kwargs['desc'] = 'progress'
    if 'mininterval' not in kwargs:
        kwargs['mininterval'] = 1
    kwargs['leave'] = False
    yield tqdm.tqdm(file=TqdmLogger(logger), **kwargs)
