# -*- coding: utf-8 -*-
import unittest

from buffalo.misc import aux
from buffalo.algo.als import ALS
from buffalo.algo.options import ALSOption
from buffalo.misc.log import set_log_level
from buffalo.data.mm import MatrixMarketOptions

from .base import TestBase, MockAlgo


class TestAlgoBase(TestBase):
    def test00_tensorboard(self):
        set_log_level(2)
        opt = ALSOption().get_default_option()
        opt.d = 5
        opt.validation = aux.Option({'topk': 10})
        opt.tensorboard = aux.Option({'root': './tb',
                                      'name': 'als'})

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        als = ALS(opt, data_opt=data_opt)
        als.initialize()
        als.train()
        results = als.get_validation_results()
        self.assertTrue(results['ndcg'] > 0.025)
        self.assertTrue(results['map'] > 0.015)

    def test01_early_stopping(self):
        set_log_level(2)
        algo = MockAlgo()
        algo.initialize()
        algo.set_losses([1.0 + i / 1.0 for i in range(100)])
        algo.opt.early_stopping_rounds = 5
        algo.train()
        self.assertEqual(algo.last_iteration, 5)

    def test02_most_similar(self):
        set_log_level(2)
        opt = ALSOption().get_default_option()

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'

        als = ALS(opt, data_opt=data_opt)
        als.initialize()
        als.train()
        q1, q2, q3 = '49.Star_Wars_(1977)', '180.Return_of_the_Jedi_(1983)', '171.Empire_Strikes_Back,_The_(1980)'
        self._test_most_similar(als, q1, q2, q3)

if __name__ == '__main__':
    unittest.main()
