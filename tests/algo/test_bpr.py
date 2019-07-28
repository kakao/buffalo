# -*- coding: utf-8 -*-
import unittest
from buffalo.misc import aux
from buffalo.algo.bpr import BPRMF
from buffalo.misc.log import set_log_level
from buffalo.algo.options import BprmfOption
from buffalo.data.mm import MatrixMarketOptions

from .base import TestBase


class TestBPRMF(TestBase):
    def test0_get_default_option(self):
        BprmfOption().get_default_option()
        self.assertTrue(True)

    def test1_is_valid_option(self):
        # TODO
        return
        """
        opt = bprOption().get_default_option()
        self.assertTrue(bprOption().is_valid_option(opt))
        opt['save_best_only'] = 1
        self.assertRaises(RuntimeError, bprOption().is_valid_option, opt)
        opt['save_best_only'] = Fbpre
        self.assertTrue(bprOption().is_valid_option(opt))"""

    def test2_init_with_dict(self):
        set_log_level(3)
        opt = BprmfOption().get_default_option()
        BPRMF(opt)
        self.assertTrue(True)

    def test3_init(self):
        set_log_level(3)
        opt = BprmfOption().get_default_option()
        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.path = './ml100k.h5py'

        bpr = BPRMF(opt, data_opt=data_opt)
        self.assertTrue(True)
        bpr.initialize()
        self.assertTrue(bpr.P.shape, (943, 20))
        self.assertTrue(bpr.Q.shape, (1682, 20))
        self.assertTrue(bpr.Qb.shape, (1682, 1))

    def test4_train(self):
        set_log_level(3)
        opt = BprmfOption().get_default_option()
        opt.d = 5

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        bpr = BPRMF(opt, data_opt=data_opt)
        bpr.initialize()
        bpr.train()
        self.assertTrue(True)

    def test5_validation(self):
        set_log_level(2)
        opt = BprmfOption().get_default_option()
        opt.d = 5
        opt.lr = 0.001
        opt.sampling_power = 0.0
        opt.num_iters = 1000
        opt.validation = aux.Option({'topk': 10})

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        bpr = BPRMF(opt, data_opt=data_opt)
        bpr.initialize()
        bpr.train()
        results = bpr.get_validation_results()
        print(results)


if __name__ == '__main__':
    unittest.main()
