# -*- coding: utf-8 -*-
import unittest

import numpy as np

from buffalo.misc import aux
from buffalo.misc.log import set_log_level
from buffalo.algo.options import BPRMFOption
from buffalo.algo.bpr import BPRMF, inited_CUBPR

from .base import TestBase


class TestBPRMF(TestBase):
    def test00_get_default_option(self):
        BPRMFOption().get_default_option()
        self.assertTrue(True)

    def test01_is_valid_option(self):
        opt = BPRMFOption().get_default_option()
        self.assertTrue(BPRMFOption().is_valid_option(opt))
        opt['save_best'] = 1
        self.assertRaises(RuntimeError, BPRMFOption().is_valid_option, opt)
        opt['save_best'] = False
        self.assertTrue(BPRMFOption().is_valid_option(opt))

    def test02_init_with_dict(self):
        set_log_level(3)
        opt = BPRMFOption().get_default_option()
        BPRMF(opt)
        self.assertTrue(True)

    def test03_init(self):
        opt = BPRMFOption().get_default_option()
        self._test3_init(BPRMF, opt)

    def test04_train(self):
        opt = BPRMFOption().get_default_option()
        opt.d = 5
        self._test4_train(BPRMF, opt)

    def test05_validation(self):
        np.random.seed(7)
        opt = BPRMFOption().get_default_option()
        opt.d = 5
        opt.num_workers = 4
        opt.num_iters = 500
        opt.random_seed = 7
        opt.validation = aux.Option({'topk': 10})
        opt.tensorboard = aux.Option({'root': './tb',
                                      'name': 'bpr'})

        self._test5_validation(BPRMF, opt, ndcg=0.03, map=0.02)

    def test06_topk(self):
        opt = BPRMFOption().get_default_option()
        opt.d = 5
        opt.num_iters = 200
        opt.validation = aux.Option({'topk': 10})
        self._test6_topk(BPRMF, opt)

    def test07_train_ml_20m(self):
        opt = BPRMFOption().get_default_option()
        opt.num_workers = 8
        opt.validation = aux.Option({'topk': 10})
        self._test7_train_ml_20m(BPRMF, opt)

    def test08_serialization(self):
        opt = BPRMFOption().get_default_option()
        opt.num_iters = 200
        opt.d = 5
        opt.validation = aux.Option({'topk': 10})

        self._test8_serialization(BPRMF, opt)

    def test09_compact_serialization(self):
        opt = BPRMFOption().get_default_option()
        opt.num_iters = 200
        opt.d = 5
        opt.validation = aux.Option({'topk': 10})
        self._test9_compact_serialization(BPRMF, opt)

    def test10_fast_most_similar(self):
        opt = BPRMFOption().get_default_option()
        opt.d = 5
        opt.validation = aux.Option({'topk': 10})
        self._test10_fast_most_similar(BPRMF, opt)

    def test11_gpu_validation(self):
        if not inited_CUBPR:
            return
        np.random.seed(7)
        opt = BPRMFOption().get_default_option()
        opt.d = 100
        opt.verify_neg = False
        opt.accelerator = True
        opt.lr = 0.01
        opt.reg_b = 10.0
        opt.num_iters = 500
        opt.evaluation_period = 50
        opt.random_seed = 777
        opt.validation = aux.Option({'topk': 10})
        opt.tensorboard = aux.Option({'root': './tb',
                                      'name': 'bpr'})

        self._test5_validation(BPRMF, opt, ndcg=0.03, map=0.02)

    def test12_gpu_train_ml_20m(self):
        if not inited_CUBPR:
            return
        opt = BPRMFOption().get_default_option()
        opt.accelerator = True
        opt.d = 100
        opt.verify_neg = False
        opt.num_iters = 30
        opt.evaluation_period = 5
        opt.validation = aux.Option({'topk': 10})
        self._test7_train_ml_20m(BPRMF, opt)


if __name__ == '__main__':
    unittest.main()
