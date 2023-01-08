# -*- coding: utf-8 -*-
import os
import unittest

import numpy as np

import buffalo.data
from buffalo import aux, set_log_level
from buffalo import PLSI, PLSIOption
from buffalo import MatrixMarketOptions

from .base import TestBase


class TestPLSI(TestBase):
    def test00_get_default_option(self):
        set_log_level(3)
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

    def test05_1_validation_with_callback(self):
        opt = PLSIOption().get_default_option()
        opt.validation = aux.Option({'topk': 10})
        self._test5_1_validation_with_callback(PLSI, opt)

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

    def test10_inheritance(self):
        uid1 = np.arange(0, 100).astype(str)
        iid1 = np.arange(200, 300).astype(str)
        main1 = np.random.binomial(1, 0.1, size=(100, 100))
        data_opt1 = MatrixMarketOptions().get_default_option()
        data_opt1.input = aux.Option({'main': main1, 'uid': uid1, 'iid': iid1})
        data1 = buffalo.data.load(data_opt1)
        data1.create()
        plsi_opt1 = PLSIOption().get_default_option()
        model1 = PLSI(plsi_opt1, data=data1)
        model1.initialize()
        model1.train()
        model1.save('inherit_model.plsi')
        os.remove('mm.h5py')

        uid2 = np.arange(80, 150).astype(str)
        iid2 = np.arange(280, 350).astype(str)
        main2 = np.random.binomial(1, 0.1, size=(70, 70))
        data_opt2 = MatrixMarketOptions().get_default_option()
        data_opt2.input = aux.Option({'main': main2, 'uid': uid2, 'iid': iid2})
        data2 = buffalo.data.load(data_opt2)
        data2.create()
        plsi_opt2 = PLSIOption().get_default_option()
        plsi_opt2.inherit_opt = aux.Option({'model_path': 'inherit_model.plsi', 'inherit_user': True, 'inherit_item': True})  # noqa: E501
        model2 = PLSI(plsi_opt2, data=data2)
        model2.initialize()
        os.remove('mm.h5py')
        os.remove('inherit_model.plsi')

        prev_p, curr_p = model1.P[-20:], model2.P[:20]
        prev_q, curr_q = model1.Q[-20:], model2.Q[:20]
        self.assertTrue(np.allclose(prev_p, curr_p))
        self.assertTrue(np.allclose(prev_q, curr_q))


if __name__ == '__main__':
    unittest.main()
