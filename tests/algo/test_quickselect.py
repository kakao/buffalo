# -*- coding: utf-8 -*-
import os
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

class TestQuickSelect(TestBase):
    def test0_np_argsort(self):
        st = time.time()
        res = np.argsort(-scores)[:, :topk]
        el = time.time() - st
        print(f"res: {res[0]}, el: {el} sec")

    def test1_quickselect(self):
        ev = Evaluable()
        st = time.time()
        res = ev.get_topk(scores, k=topk, num_threads=4)
        el = time.time() - st
        print(f"res: {res[0]}, el: {el} sec")


if __name__ == '__main__':
    unittest.main()
