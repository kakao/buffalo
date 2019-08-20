# -*- coding: utf-8 -*-
import time

import unittest
import implicit
from scipy import io as sio

from .base import TestBase
from buffalo.misc import log


class TestImplicit(TestBase):
    def test0_train_ml20m_cpu(self):
        logger = log.get_logger("implicit")
        num_iters = 10
        model = implicit.als.AlternatingLeastSquares(factors=128, regularization=0.1,
                                                     use_gpu=False, iterations=num_iters,
                                                     num_threads=8)
        mm_path = "ml-20m/main"
        logger.info(f"read matrix from {mm_path}")
        st = time.time()
        user_item_data = sio.mmread(mm_path)
        el, st = time.time() - st, time.time()
        logger.info(f"elasped for reading data: {el} sec")

        model.fit(user_item_data)
        el, st = time.time() - st, time.time()
        logger.info(f"elasped for {num_iters} iterations: {el} sec")

    def test1_train_ml20m_gpu(self):
        logger = log.get_logger("implicit")
        num_iters = 10
        model = implicit.als.AlternatingLeastSquares(factors=128, regularization=0.1,
                                                     use_gpu=True, iterations=num_iters)
        mm_path = "ml-20m/main"
        logger.info(f"read matrix from {mm_path}")
        st = time.time()
        user_item_data = sio.mmread(mm_path)
        el, st = time.time() - st, time.time()
        logger.info(f"elasped for reading data: {el} sec")

        model.fit(user_item_data)
        el, st = time.time() - st, time.time()
        logger.info(f"elasped for {num_iters} iterations: {el} sec")


if __name__ == '__main__':
    unittest.main()
