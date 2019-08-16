# -*- coding: utf-8 -*-
import os
import unittest

from hyperopt import fmin, tpe

from buffalo.misc import aux
from buffalo.algo.als import ALS
from buffalo.misc.log import set_log_level
from buffalo.algo.options import AlsOption
from buffalo.algo.optimize import Optimizable
from buffalo.data.mm import MatrixMarketOptions

from .base import TestBase, MockAlgo


class TestOptimize(TestBase):
    def test0_get_space(self):
        Optimizable()._get_space({'x': ['uniform', ['x', 1, 10]]})
        self.assertTrue(True)

    def test1_optimize(self):
        space = Optimizable()._get_space({'x': ['uniform', ['x', 1, 10]]})
        best = fmin(fn=lambda opt: opt['x'] ** 2,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=100)
        self.assertEqual(int(best['x']), 1)

    def test2_optimize(self):
        def mock_fn(opt):
            loss = 1.0 - opt['adaptive_reg'] / 1.0
            loss += 1.0 / opt['d']
            loss += 1.0 / opt['alpha']
            loss += 1.0 / opt['reg_i']
            loss += 1.0 / opt['reg_u']
            return loss

        option = AlsOption().get_default_optimize_option()
        space = Optimizable()._get_space(option.space)
        best = fmin(fn=mock_fn,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=600)
        self.assertGreaterEqual(int(best['d']), 16)
        self.assertGreaterEqual(int(best['alpha']), 16)
        self.assertGreaterEqual(best['reg_i'], 0.5)
        self.assertGreaterEqual(best['reg_u'], 0.5)
        self.assertEqual(best['adaptive_reg'], 1)

    def test3_optimize(self):
        algo = MockAlgo()
        algo.optimize()
        self.assertTrue(True)

    def test4_optimize(self):
        set_log_level(2)
        opt = AlsOption().get_default_option()
        opt.d = 5
        opt.num_workers = 2
        opt.model_path = 'als.bin'
        opt.validation = aux.Option({'topk': 10})
        optimize_option = aux.Option({
            'loss': 'val_rmse',
            'max_trials': 10,
            'deployment': True,
            'start_with_default_parameters': True,
            'space': {
                'd': ['randint', ['d', 10, 20]],
                'reg_u': ['uniform', ['reg_u', 0.1, 0.3]],
                'reg_i': ['uniform', ['reg_i', 0.1, 0.3]],
                'alpha': ['randint', ['alpha', 8, 10]]
            }
        })
        opt.optimize = optimize_option
        opt.evaluation_period = 1
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
        default_result = als.get_validation_results()
        als.optimize()
        base_loss = default_result['rmse']  # val_rmse
        optimize_loss = als.get_optimization_data()['best']['val_rmse']
        self.assertTrue(base_loss > optimize_loss)

        als.load('als.bin')
        loss = als.get_validation_results()
        self.assertAlmostEqual(loss['rmse'], optimize_loss)
        os.remove('als.bin')


if __name__ == '__main__':
    unittest.main()
