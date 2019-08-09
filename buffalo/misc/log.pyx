# cython: experimental_cpp_class_def=True
# distutils: language=c++
# -*- coding: utf-8 -*-
import io
import os
import sys
import logging
import logging.handlers
from contextlib import contextmanager

import tqdm

NOTSET = 0
WARN = 1
INFO = 2
DEBUG = 3
TRACE = 4
__logger_holder = []


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
    for logger in __logger_holder:
        if lvl == NOTSET:
            logger.setLevel(logging.NOTSET)
        elif lvl == WARN:
            logger.setLevel(logging.WARN)
        elif lvl == INFO:
            logger.setLevel(logging.INFO)
        elif lvl == DEBUG:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.DEBUG)
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
            self.buf = ''


def get_logger(name=__file__, no_fileno=False):
    global __logger_holder
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    lvl = get_log_level()
    if lvl == NOTSET:
        logger.setLevel(logging.NOTSET)
    elif lvl == WARN:
        logger.setLevel(logging.WARN)
    elif lvl == INFO:
        logger.setLevel(logging.INFO)
    elif lvl == DEBUG:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    if no_fileno:
        formatter = logging.Formatter('[%(levelname)-8s] %(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
    else:
        formatter = logging.Formatter('[%(levelname)-8s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S')
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    __logger_holder.append(logger)
    return logger


@contextmanager
def pbar(log_level=INFO, **kwargs):
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
    logger = get_logger('pbar', no_fileno=True)
    logger_func = logger.info
    if log_level == INFO:
        logger_func = logger.info
    elif log_level == WARN:
        logger_func = logger.warn
    elif log_level == DEBUG:
        logger_func = logger.debug
    yield tqdm.tqdm(file=TqdmLogger(logger_func), **kwargs)


def iter_pbar(log_level=INFO, **kwargs):
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
    logger = get_logger('pbar', no_fileno=True)
    logger_func = logger.info
    if log_level == INFO:
        logger_func = logger.info
    elif log_level == WARN:
        logger_func = logger.warn
    elif log_level == DEBUG:
        logger_func = logger.debug
    return tqdm.tqdm(file=TqdmLogger(logger_func), **kwargs)


class supress_log_level(object):
    def __init__(self, log_level):
        self.desire_log_level = log_level
        self.original_log_level = get_log_level()

    def __enter__(self):
        set_log_level(self.desire_log_level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_log_level(self.original_log_level)
