# -*- coding: utf-8 -*-
import unittest

from buffalo.misc import aux
from buffalo.algo.plsi import PLSI
from buffalo.algo.options import PLSIOption

from .base import TestBase


class TestPLSI(TestBase):

    def test00_get_default_option(self):
        PLSIOption().get_default_option()
        self.assertTrue(True)

    def test01_is_valid_option(self):
        opt = PLSIOption().get_default_option()
        self.assertTrue(PLSIOption().is_valid_option(opt))
        opt['save_best'] = 1
        self.assertRaises(RuntimeError, PLSIOption().is_valid_option, opt)
        opt['save_best'] = False
        self.assertTrue(PLSIOption().is_valid_option(opt))

    def test02_init_with_dict(self):
        opt = PLSIOption().get_default_option()
        PLSI(opt)
        self.assertTrue(True)

    def test03_init(self):
        opt = PLSIOption().get_default_option()
        self._test3_init(PLSI, opt)

    def test04_train(self):
        opt = PLSIOption().get_default_option()
        self._test4_train(PLSI, opt)

    def test05_validation(self):
        opt = PLSIOption().get_default_option()
        opt.validation = aux.Option({'topk': 10})
        self._test5_validation(PLSI, opt, ndcg=0.03, map=0.02)

    def test06_topk(self):
        opt = PLSIOption().get_default_option()
        opt.d = 10
        self.maxDiff = None
        self._test6_topk(PLSI, opt)

    def test07_train_ml_20m(self):
        opt = PLSIOption().get_default_option()
        opt.num_workers = 8
        self._test7_train_ml_20m(PLSI, opt)

    def test08_serialization(self):
        opt = PLSIOption().get_default_option()
        opt.d = 10
        self._test8_serialization(PLSI, opt)

    def test09_compact_serialization(self):
        opt = PLSIOption().get_default_option()
        opt.d = 10
        self._test9_compact_serialization(PLSI, opt)


if __name__ == '__main__':
    unittest.main()
