# -*- coding: utf-8 -*-
import os
import time
import logging
import unittest
logging.getLogger('tensorflow').disabled = True

import numpy as np
from hyperopt import STATUS_OK

from buffalo.misc import aux, log
from buffalo.algo.base import Algo
from buffalo.misc.log import set_log_level
from buffalo.algo.options import ALSOption
from buffalo.algo.optimize import Optimizable
from buffalo.data.mm import MatrixMarketOptions
from buffalo.algo.base import TensorboardExtention


class MockAlgo(Algo, Optimizable, TensorboardExtention):
    def __init__(self, *args, **kwargs):
        Algo.__init__(self, *args, **kwargs)
        Optimizable.__init__(self, *args, **kwargs)
        TensorboardExtention.__init__(self, *args, **kwargs)
        self.logger = log.get_logger('MockAlgo')
        option = ALSOption().get_default_option()
        optimize_option = ALSOption().get_default_optimize_option()
        optimize_option.start_with_default_parameters = False
        option.optimize = optimize_option
        option.model_path = 'hello.world.bin'
        self.opt = option
        self._optimize_loss = {'loss': 987654321.0}

    def _optimize(self, params):
        self._optimize_params = params
        loss = 1.0 - params['adaptive_reg'] / 1.0
        loss += 1.0 / params['d']
        loss += 1.0 / params['alpha']
        loss += 1.0 / params['reg_i']
        loss += 1.0 / params['reg_u']
        self.validation_result = {'loss': loss}
        return {'loss': loss,
                'status': STATUS_OK}

    def save(self, path):
        return path

    def _get_feature(self, index, group='item'):
        pass

    def normalize(self, group='item'):
        pass

    def set_losses(self, losses):
        self.losses = losses

    def train(self):
        self.initialize()
        self.last_iteration = 0
        for i in range(self.opt.num_iters):
            loss = self.losses[i % len(self.losses)]
            self.last_iteration = i
            if self.early_stopping(loss):
                break


class TestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ml_100k = './ext/ml-100k/'
        cls.ml_20m = './ext/ml-20m/'
        cls.text8 = './ext/text8/'
        cls.temp_files = []

    @classmethod
    def tearDownClass(cls):
        for path in cls.temp_files:
            os.remove(path)

    def _test3_init(self, cls, opt):
        set_log_level(3)
        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.path = './ml100k.h5py'

        c = cls(opt, data_opt=data_opt)
        self.assertTrue(True)
        c.init_factors()
        self.assertEqual(c.P.shape, (943, 20))
        self.assertEqual(c.Q.shape, (1682, 20))

    def _test4_train(self, cls, opt):
        set_log_level(3)
        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        c = cls(opt, data_opt=data_opt)
        c.initialize()
        c.train()
        self.assertTrue(True)

    def _test5_validation(self, cls, opt, ndcg=0.06, map=0.04):
        set_log_level(2)

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        c = cls(opt, data_opt=data_opt)
        c.initialize()
        c.train()
        results = c.get_validation_results()
        self.assertTrue(results['ndcg'] > ndcg, msg='NDCG Test')
        self.assertTrue(results['map'] > map, msg='MAP Test')

    def _test6_topk(self, cls, opt):
        set_log_level(2)

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        c = cls(opt, data_opt=data_opt)
        c.initialize()
        c.train()
        self.assertTrue(len(c.topk_recommendation('1', 10)), 10)
        ret_a = [x for x, _ in c.most_similar('180.Return_of_the_Jedi_(1983)', topk=100)]
        self.assertIn('49.Star_Wars_(1977)', ret_a)
        c.normalize()
        ret_b = [x for x, _ in c.most_similar('180.Return_of_the_Jedi_(1983)', topk=100)]
        self.assertIn('49.Star_Wars_(1977)', ret_b)
        self.assertEqual(ret_a[:10], ret_b[:10])

    def _test7_train_ml_20m(self, cls, opt):
        set_log_level(3)

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_20m + 'main'
        data_opt.input.uid = self.ml_20m + 'uid'
        data_opt.input.iid = self.ml_20m + 'iid'
        data_opt.data.path = './ml20m.h5py'
        data_opt.data.use_cache = True

        c = cls(opt, data_opt=data_opt)
        c.initialize()
        c.train()
        self.assertTrue(True)

    def _test8_serialization(self, cls, opt):
        set_log_level(1)

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        c = cls(opt, data_opt=data_opt)
        c.initialize()
        c.train()
        ret_a = [x for x, _ in c.most_similar('180.Return_of_the_Jedi_(1983)', topk=100)]
        self.assertIn('49.Star_Wars_(1977)', ret_a)
        c.save('model.bin')
        c.load('model.bin')
        os.remove('model.bin')
        ret_a = [x for x, _ in c.most_similar('180.Return_of_the_Jedi_(1983)', topk=100)]
        self.assertIn('49.Star_Wars_(1977)', ret_a)

    def _test9_compact_serialization(self, cls, opt):
        set_log_level(1)

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        c = cls(opt, data_opt=data_opt)
        c.initialize()
        c.train()
        ret_a = [x for x, _ in c.most_similar('180.Return_of_the_Jedi_(1983)', topk=100)]
        self.assertIn('49.Star_Wars_(1977)', ret_a)
        c.save('model.bin', with_userid_map=False)
        c = cls(opt)
        c.load('model.bin', data_fields=['Q', '_idmanager'])
        ret_a = [x for x, _ in c.most_similar('180.Return_of_the_Jedi_(1983)', topk=100)]
        self.assertIn('49.Star_Wars_(1977)', ret_a)
        self.assertFalse(hasattr(c, 'P'))
        c.normalize(group='item')
        ret_a = [x for x, _ in c.most_similar('180.Return_of_the_Jedi_(1983)', topk=100)]
        self.assertIn('49.Star_Wars_(1977)', ret_a)

    def _test10_fast_most_similar(self, cls, opt):
        set_log_level(1)

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        c = cls(opt, data_opt=data_opt)
        c.initialize()
        c.train()

        keys = [x for x, _ in c.most_similar('49.Star_Wars_(1977)', topk=100)]
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

    def _test_most_similar(self, model, q1, q2, q3):
        self.assertEqual(len(model.most_similar(q1, pool=[q2])), 1)

        index = model.get_index(q2)
        ret = model.most_similar(q1, pool=np.array([index]))
        self.assertEqual(ret[0][0], q2)

        pool = [q2, q3]
        ret_a = model.most_similar(q1, pool=pool)
        indexes = model.get_index(pool)
        self.assertEqual(len(indexes), 2)
        ret_b = model.most_similar(q1, pool=indexes)
        self.assertEqual(ret_a, ret_b)

        keys = [k[0] for k in model.most_similar(q1, topk=100)]
        keys += ['fake_key', 10]
        indexes = model.get_index(keys)
        self.assertEqual(len(keys), len(indexes))
        indexes = np.array([i for i in indexes if i is not None])
        self.assertEqual(len(indexes), 100)

        start_t = time.time()
        for i in range(100):
            model.most_similar(q1, pool=keys)
        elapsed_a = time.time() - start_t

        start_t = time.time()
        for i in range(100):
            model.most_similar(q1, pool=indexes)
        elapsed_b = time.time() - start_t
        self.assertTrue(elapsed_a > elapsed_b)
