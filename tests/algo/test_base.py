# -*- coding: utf-8 -*-
import unittest
from buffalo.misc import aux
from buffalo.algo.als import ALS
from buffalo.algo.options import AlsOption
from buffalo.misc.log import set_log_level
from buffalo.data.mm import MatrixMarketOptions

from .base import TestBase


class TestAlgoBase(TestBase):
    def test0_tensorboard(self):
        set_log_level(2)
        opt = AlsOption().get_default_option()
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
        als.init_factors()
        als.train()
        results = als.get_validation_results()
        self.assertTrue(results['ndcg'] > 0.025)
        self.assertTrue(results['map'] > 0.015)

if __name__ == '__main__':
    unittest.main()
