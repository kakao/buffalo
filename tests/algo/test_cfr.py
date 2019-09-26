# -*- coding: utf-8 -*-
import os
import time

import unittest
from .base import TestBase
from buffalo.misc import aux
from buffalo.algo.cfr import CFR
from buffalo.misc.log import set_log_level
from buffalo.algo.options import CFROption
from buffalo.data.stream import StreamOptions


class TestCFR(TestBase):
    def test00_get_default_option(self):
        CFROption().get_default_option()
        self.assertTrue(True)

    def test01_is_valid_option(self):
        opt = CFROption().get_default_option()
        self.assertTrue(CFROption().is_valid_option(opt))
        opt['save_best'] = 1
        self.assertRaises(RuntimeError, CFROption().is_valid_option, opt)
        opt['save_best'] = False
        self.assertTrue(CFROption().is_valid_option(opt))

    def test02_init_with_dict(self):
        set_log_level(3)
        opt = CFROption().get_default_option()
        CFR(opt)
        self.assertTrue(True)

    def test03_init(self):
        set_log_level(3)
        opt = CFROption().get_default_option()
        opt.d = 20
        data_opt = StreamOptions().get_default_option()
        data_opt.data.sppmi = {"windows": 5, "k": 10}
        data_opt.data.internal_data_type = "matrix"
        data_opt.input.main = self.ml_100k + 'stream'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.path = './ml100k.h5py'

        c = CFR(opt, data_opt=data_opt)
        self.assertTrue(True)
        c.initialize()
        self.assertEqual(c.U.shape, (943, 20))
        self.assertEqual(c.I.shape, (1682, 20))

    def test04_train(self):
        set_log_level(3)
        opt = CFROption().get_default_option()
        data_opt = StreamOptions().get_default_option()
        data_opt.data.sppmi = {"windows": 5, "k": 10}
        data_opt.data.internal_data_type = "matrix"
        data_opt.input.main = self.ml_100k + 'stream'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        c = CFR(opt, data_opt=data_opt)
        c.initialize()
        c.train()
        self.assertTrue(True)

    def test05_validation(self, ndcg=0.06, map=0.04):
        set_log_level(3)
        opt = CFROption().get_default_option()
        opt.validation = aux.Option({'topk': 10})
        opt.tensorboard = aux.Option({'root': './tb',
                                      'name': 'cfr'})
        data_opt = StreamOptions().get_default_option()
        data_opt.data.validation.name = "sample"
        data_opt.data.sppmi = {"windows": 5, "k": 10}
        data_opt.data.internal_data_type = "matrix"
        data_opt.input.main = self.ml_100k + 'stream'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        c = CFR(opt, data_opt=data_opt)
        c.initialize()
        c.train()
        results = c.get_validation_results()
        self.assertTrue(results['ndcg'] > ndcg)
        self.assertTrue(results['map'] > map)

    def test06_topk(self):
        set_log_level(1)
        opt = CFROption().get_default_option()
        opt.validation = aux.Option({'topk': 10})
        data_opt = StreamOptions().get_default_option()
        data_opt.data.validation.name = "sample"
        data_opt.data.sppmi = {"windows": 5, "k": 10}
        data_opt.data.internal_data_type = "matrix"
        data_opt.input.main = self.ml_100k + 'stream'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        c = CFR(opt, data_opt=data_opt)
        c.initialize()
        c.train()
        self.assertTrue(len(c.topk_recommendation('1', 10)), 10)
        ret_a = [x for x, _ in c.most_similar('49.Star_Wars_(1977)')]
        self.assertIn('180.Return_of_the_Jedi_(1983)', ret_a)
        c.normalize()
        ret_b = [x for x, _ in c.most_similar('49.Star_Wars_(1977)')]
        self.assertIn('180.Return_of_the_Jedi_(1983)', ret_b)
        self.assertEqual(ret_a, ret_b)

    def test07_train_ml_20m(self):
        set_log_level(3)
        opt = CFROption().get_default_option()
        data_opt = StreamOptions().get_default_option()
        data_opt.data.sppmi = {"windows": 5, "k": 10}
        data_opt.data.internal_data_type = "matrix"
        data_opt.data.tmp_dir = './tmp/'
        data_opt.input.main = self.ml_20m + 'stream'
        data_opt.input.uid = self.ml_20m + 'uid'
        data_opt.input.iid = self.ml_20m + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        c = CFR(opt, data_opt=data_opt)
        c.initialize()
        c.train()
        self.assertTrue(True)

    def test08_serialization(self):
        set_log_level(1)

        opt = CFROption().get_default_option()
        data_opt = StreamOptions().get_default_option()
        data_opt.data.sppmi = {"windows": 5, "k": 10}
        data_opt.data.internal_data_type = "matrix"
        data_opt.input.main = self.ml_100k + 'stream'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        c = CFR(opt, data_opt=data_opt)
        c.initialize()
        c.train()
        ret_a = [x for x, _ in c.most_similar('49.Star_Wars_(1977)')]
        self.assertIn('180.Return_of_the_Jedi_(1983)', ret_a)
        c.save('model.bin')
        c.load('model.bin')
        os.remove('model.bin')
        ret_a = [x for x, _ in c.most_similar('49.Star_Wars_(1977)')]
        self.assertIn('180.Return_of_the_Jedi_(1983)', ret_a)

    def test09_compact_serialization(self):
        set_log_level(1)

        opt = CFROption().get_default_option()
        data_opt = StreamOptions().get_default_option()
        data_opt.data.sppmi = {"windows": 5, "k": 10}
        data_opt.data.internal_data_type = "matrix"
        data_opt.input.main = self.ml_100k + 'stream'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        c = CFR(opt, data_opt=data_opt)
        c.initialize()
        c.train()
        ret_a = [x for x, _ in c.most_similar('49.Star_Wars_(1977)')]
        self.assertIn('180.Return_of_the_Jedi_(1983)', ret_a)
        c.save('model.bin', with_userid_map=False)
        c = CFR(opt)
        c.load('model.bin', data_fields=['I', '_idmanager'])
        ret_a = [x for x, _ in c.most_similar('49.Star_Wars_(1977)')]
        self.assertIn('180.Return_of_the_Jedi_(1983)', ret_a)
        self.assertFalse(hasattr(c, 'U'))
        c.normalize(group='item')
        ret_a = [x for x, _ in c.most_similar('49.Star_Wars_(1977)')]
        self.assertIn('180.Return_of_the_Jedi_(1983)', ret_a)

    def test10_fast_most_similar(self):
        set_log_level(1)

        opt = CFROption().get_default_option()
        data_opt = StreamOptions().get_default_option()
        data_opt.data.sppmi = {"windows": 5, "k": 10}
        data_opt.data.internal_data_type = "matrix"
        data_opt.input.main = self.ml_100k + 'stream'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        c = CFR(opt, data_opt=data_opt)
        c.initialize()
        c.train()

        keys = [x for x, _ in c.most_similar('49.Star_Wars_(1977)', 10)]
        start_t = time.time()
        for i in range(100):
            for key in keys:
                c.most_similar(key)
        elapsed_a = time.time() - start_t

        c.normalize(group='item')
        start_t = time.time()
        for i in range(100):
            for key in keys:
                c.most_similar(key)
        elapsed_b = time.time() - start_t
        self.assertTrue(elapsed_a > elapsed_b)


if __name__ == '__main__':
    unittest.main()
