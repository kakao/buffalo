# -*- coding: utf-8 -*-
import os
import unittest
from buffalo.misc import aux
from buffalo.algo.als import ALS
from buffalo.algo.options import AlsOption
from buffalo.data.mm import MatrixMarketOptions
from buffalo.misc.log import set_log_level, get_log_level

from .base import TestBase


class TestALS(TestBase):
    def test0_get_default_option(self):
        AlsOption().get_default_option()
        self.assertTrue(True)

    def test1_is_valid_option(self):
        opt = AlsOption().get_default_option()
        self.assertTrue(AlsOption().is_valid_option(opt))
        opt['save_best_only'] = 1
        self.assertRaises(RuntimeError, AlsOption().is_valid_option, opt)
        opt['save_best_only'] = False
        self.assertTrue(AlsOption().is_valid_option(opt))

    def test2_init_with_dict(self):
        opt = AlsOption().get_default_option()
        ALS(opt)
        self.assertTrue(True)

    def test3_init(self):
        set_log_level(3)
        opt = AlsOption().get_default_option()
        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.path = './ml100k.h5py'

        als = ALS(opt, data_opt=data_opt)
        self.assertTrue(True)
        als.init_factors()
        self.assertTrue(als.P.shape, (5, 20))
        self.assertTrue(als.Q.shape, (3, 20))

    def test4_train(self):
        set_log_level(3)
        opt = AlsOption().get_default_option()
        opt.d = 5

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        als = ALS(opt, data_opt=data_opt)
        als.init_factors()
        als.train()

    def test5_validation(self):
        set_log_level(1)
        opt = AlsOption().get_default_option()
        opt.d = 5
        opt.validation = aux.Option({'topk': 10})

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        als = ALS(opt, data_opt=data_opt)
        als.init_factors()
        als.train()
        results = als.get_validation_results()
        self.assertTrue(results['ndcg'] > 0.03)
        self.assertTrue(results['map'] > 0.015)

    def test6_topk(self):
        set_log_level(1)
        opt = AlsOption().get_default_option()
        opt.d = 5
        opt.validation = aux.Option({'topk': 10})

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        als = ALS(opt, data_opt=data_opt)
        als.init_factors()
        als.train()
        als.data.build_idmaps()
        self.assertTrue(len(als.topk_recommendation('1', 10)['1']), 10)
        als.fast_similar(True)
        ret_a = als.most_similar('Star_Wars_(1977)', 10)['Star_Wars_(1977)']
        self.assertIn('Return_of_the_Jedi_(1983)', ret_a)
        als.fast_similar(False)
        ret_b = als.most_similar('Star_Wars_(1977)', 10)['Star_Wars_(1977)']
        self.assertIn('Return_of_the_Jedi_(1983)', ret_b)
        self.assertEqual(ret_a, ret_b)

    def test7_train_ml_20m(self):
        set_log_level(2)
        opt = AlsOption().get_default_option()
        opt.num_workers = 8
        opt.validation = aux.Option({'topk': 10})

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_20m + 'main'
        data_opt.input.uid = self.ml_20m + 'uid'
        data_opt.input.iid = self.ml_20m + 'iid'
        data_opt.data.path = './ml20m.h5py'
        data_opt.data.use_cache = True

        als = ALS(opt, data_opt=data_opt)
        als.init_factors()
        als.train()

if __name__ == '__main__':
    unittest.main()
