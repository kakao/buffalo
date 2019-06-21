# -*- coding: utf-8 -*-
import unittest
from buffalo.algo.als import ALS
from buffalo.algo.options import AlsOption
from buffalo.misc.log import set_log_level
from buffalo.data import MatrixMarketOptions

from .base import TestBase


class TestALS(TestBase):
    def test0_get_default_option(self):
        AlsOption().get_default_option()
        self.assertTrue(True)

    def test0_is_valid_option(self):
        opt = AlsOption().get_default_option()
        self.assertTrue(AlsOption().is_valid_option(opt))
        opt['save_best_only'] = 1
        self.assertRaises(RuntimeError, AlsOption().is_valid_option, opt)
        opt['save_best_only'] = False
        self.assertTrue(AlsOption().is_valid_option(opt))

    def test1_init_with_dict(self):
        opt = AlsOption().get_default_option()
        ALS(opt)
        self.assertTrue(True)

    def test2_init(self):
        set_log_level(2)
        opt = AlsOption().get_default_option()
        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.mm_path
        data_opt.input.uid = self.uid_path
        data_opt.input.iid = self.iid_path

        als = ALS(opt, data_opt=data_opt)
        self.assertTrue(True)
        self.temp_files.append(data_opt.data.path)
        als.init_factors()
        self.assertTrue(als.P.shape, (5, 20))
        self.assertTrue(als.Q.shape, (3, 20))

    def test3_train(self):
        set_log_level(2)
        opt = AlsOption().get_default_option()
        opt.d = 5

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.mm_path
        data_opt.input.uid = self.uid_path
        data_opt.input.iid = self.iid_path

        als = ALS(opt, data_opt=data_opt)
        als.init_factors()
        als.train()


if __name__ == '__main__':
    unittest.main()
