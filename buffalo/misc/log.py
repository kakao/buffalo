# -*- coding: utf-8 -*-
import time
import logging
import logging.handlers

from buffalo.misc._log import PyBuffaloLog


NOTSET = 0
WARN = 1
INFO = 2
DEBUG = 3
TRACE = 4
__logger_holder = []

devnull = None


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
    if name == 'pbar':
        formatter = logging.Formatter('%(message)s')
    elif no_fileno:
        formatter = logging.Formatter('[%(levelname)-8s] %(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
    else:
        formatter = logging.Formatter('[%(levelname)-8s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S')
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    __logger_holder.append(logger)
    return logger


class ProgressBar(object):
    def __init__(self, level, **kwargs):
        self.lvl = level
        self._logger = get_logger('pbar', no_fileno=True)
        self.logger = self.get_logger_func(self._logger, self.lvl)
        for handle in self._logger.handlers:
            self.terminator = handle.terminator
            handle.terminator = ''
        self.initialize(**kwargs)

    def get_logger_func(self, logger, level):
        if level == INFO:
            return logger.info
        elif level == WARN:
            return logger.warn
        elif level == DEBUG:
            return logger.debug
        else:
            return logger.debug

    def initialize(self, **kwargs):
        if 'iterable' in kwargs:
            try:
                kwargs['total'] = len(kwargs['iterable'])
            except Exception:
                pass
            self.iterable = kwargs['iterable']
        if 'total' in kwargs:
            self.fmt = '\r[{desc}] {percent} {elapsed:0.1f}/{remains:0.1f}secs {ips:0,.2f}it/s'
        else:
            self.fmt = '\r[{desc}] {n_fmt:10,d}it {elapsed:0.1f}secs {ips:0,.2f}it/s'
        self.s_t = time.time()
        self.t = time.time()
        self.desc = kwargs.get('desc', 'PROGRESS')
        self.period_secs = kwargs.get('mininteral', 1)
        self.total = kwargs.get('total', -1)
        self.step = 0

    def get_msg(self):
        t = time.time()
        elapsed = t - self.s_t
        remains = 0
        percent = '0.00%'
        ips = 0.0
        if self.step > 0:
            remains = self.total / self.step * elapsed
            ips = self.step / elapsed
            percent = '%3.2f%s' % (elapsed / remains * 100, '%')
        msg = self.fmt.format(desc=self.desc,
                              percent=percent,
                              n_fmt=int(self.step),
                              total_fmt=self.total,
                              elapsed=elapsed,
                              remains=remains,
                              ips=ips)
        return msg

    def __enter__(self):
        self.logger(self.get_msg())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.total != -1:
            self.step = self.total
        self.logger(self.get_msg() + '\n')

    def __iter__(self):
        delta = None
        cnts = 0
        period_iter = 0
        for obj in self.iterable:
            yield obj
            cnts += 1
            self.step += 1
            if period_iter != 0 and (cnts % period_iter == 0):
                self.logger(self.get_msg())
            else:
                delta = time.time() - self.s_t
                if delta > self.period_secs:
                    period_iter = cnts
        if self.total != -1:
            self.step = self.total
        self.logger(self.get_msg() + '\n')

    def update(self, step):
        self.step += step
        t = time.time()
        if self.t + self.period_secs < t:
            self.t = t
            self.logger(self.get_msg())

    def refresh(self):
        # backward-compatability
        pass


class supress_log_level(object):
    def __init__(self, log_level):
        self.desire_log_level = log_level
        self.original_log_level = get_log_level()

    def __enter__(self):
        set_log_level(self.desire_log_level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_log_level(self.original_log_level)
