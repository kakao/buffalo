# -*- coding: utf-8 -*-
import unittest

import numpy as np

from buffalo.misc import aux
from buffalo.algo.warp import WARP
from buffalo.misc.log import set_log_level
from buffalo.algo.options import WARPOption

from .base import TestBase


class TestWARP(TestBase):
    def get_opts(self):
        opt_dot = WARPOption().get_default_option()
        opt_l2 = WARPOption().get_default_option()
        opt_l2['score_func'] = 'L2'
        return [opt_dot, opt_l2]

    def test00_get_default_option(self):
        WARPOption().get_default_option()
        self.assertTrue(True)

    def test01_is_valid_option(self):
        for opt in self.get_opts():
            self.assertTrue(WARPOption().is_valid_option(opt))
            opt['save_best'] = 1
            self.assertRaises(RuntimeError, WARPOption().is_valid_option, opt)
            opt['save_best'] = False
            self.assertTrue(WARPOption().is_valid_option(opt))

    def test02_init_with_dict(self):
        set_log_level(3)
        for opt in self.get_opts():
            WARP(opt)
            self.assertTrue(True)

    def test03_init(self):
        for opt in self.get_opts():
            opt.d = 20
            self._test3_init(WARP, opt)

    def test04_train(self):
        for opt in self.get_opts():
            opt.d = 32
            opt.max_tirals = 200
            self._test4_train(WARP, opt)

    def test05_validation(self):
        np.random.seed(7)
        for opt in self.get_opts():
            opt.validation = aux.Option({'topk': 10})
            # opt.tensorboard = aux.Option({'root': './tb',
            #                               'name': 'warp'})
            self._test5_validation(WARP, opt, ndcg=0.03, map=0.02)

    def test06_topk(self):
        for opt in self.get_opts():
            opt.d = 10
            opt.validation = aux.Option({'topk': 10})
            self._test6_topk(WARP, opt)

    def test07_train_ml_20m(self):
        for opt in self.get_opts():
            opt.num_workers = 8
            opt.validation = aux.Option({'topk': 10})
            self._test7_train_ml_20m(WARP, opt)

    def test08_serialization(self):
        for opt in self.get_opts():
            opt.d = 20
            opt.max_trials = 500
            opt.validation = aux.Option({'topk': 10})
            self._test8_serialization(WARP, opt)

    def test09_compact_serialization(self):
        for opt in self.get_opts():
            opt.d = 10
            opt.validation = aux.Option({'topk': 10})
            self._test9_compact_serialization(WARP, opt)


if __name__ == '__main__':
    unittest.main()
