# -*- coding: utf-8 -*-
import time
from os import environ
environ["OMP_NUM_THREADS"] = "4"
environ["OPENBLAS_NUM_THREADS"] = "4"
environ["MKL_NUM_THREADS"] = "4"
environ["VECLIB_MAXIMUM_THREADS"] = "4"
environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np

import unittest
from .base import TestBase
from buffalo.evaluate.base import Evaluable


scores = np.random.uniform(size=(100, 100000)).astype(np.float32)
topk = 10


def time_np_argsort():
    st = time.time()
    res = np.argsort(-scores)[:, :topk]
    el = time.time() - st
    return res, el


def time_np_argpartition():
    st = time.time()
    res = np.argpartition(-scores, topk)[:, :topk]
    res = np.array([sorted(row, key=lambda x:-scores[i, x]) for i, row in enumerate(res)])
    el = time.time() - st
    return res, el


def time_quickselect():
    ev = Evaluable()
    st = time.time()
    res = ev.get_topk(scores, k=topk, num_threads=4)
    el = time.time() - st
    return res, el


class TestQuickSelect(TestBase):
    def test_0_quickselect(self):
        res_argsort, t_np_argsort = time_np_argsort()
        res_argpart, t_np_argparttion = time_np_argpartition()
        res_quickselect, t_quickselect = time_quickselect()

        self.assertGreaterEqual(t_np_argsort / t_quickselect, 1)
        self.assertGreaterEqual(t_np_argparttion / t_quickselect, 1)


if __name__ == '__main__':
    unittest.main()
