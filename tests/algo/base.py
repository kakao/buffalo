# -*- coding: utf-8 -*-
import os
import time
import unittest

from buffalo.misc import aux
from buffalo.misc.log import set_log_level
from buffalo.data.mm import MatrixMarketOptions


class TestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isdir('ml-100k/'):
            raise RuntimeError('Cannot find the ./ml-100k directory')
        cls.ml_100k = './ml-100k/'

        if not os.path.isdir('ml-20m'):
            raise RuntimeError('Cannot find the ./ml-20m directory')
        cls.ml_20m = './ml-20m/'

        if not os.path.isdir('text8'):
            raise RuntimeError('Cannot find the ./text8 directory')
        cls.text8 = './text8/'
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
        self.assertTrue(len(c.topk_recommendation('1', 10)['1']), 10)
        ret_a = [x for x, _ in c.most_similar('180.Return_of_the_Jedi_(1983)', topk=100)]
        self.assertIn('229.Star_Trek_IV:_The_Voyage_Home_(1986)', ret_a)
        c.normalize()
        ret_b = [x for x, _ in c.most_similar('180.Return_of_the_Jedi_(1983)', topk=100)]
        self.assertIn('229.Star_Trek_IV:_The_Voyage_Home_(1986)', ret_b)
        self.assertEqual(ret_a[:10], ret_b[:10])

    def _test7_train_ml_20m(self, cls, opt):
        set_log_level(2)

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
        self.assertIn('229.Star_Trek_IV:_The_Voyage_Home_(1986)', ret_a)
        c.save('model.bin')
        c.load('model.bin')
        os.remove('model.bin')
        ret_a = [x for x, _ in c.most_similar('180.Return_of_the_Jedi_(1983)', topk=100)]
        self.assertIn('229.Star_Trek_IV:_The_Voyage_Home_(1986)', ret_a)

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
        self.assertIn('229.Star_Trek_IV:_The_Voyage_Home_(1986)', ret_a)
        c.save('model.bin', with_userid_map=False)
        c = cls(opt)
        c.load('model.bin', data_fields=['Q', '_idmanager'])
        ret_a = [x for x, _ in c.most_similar('180.Return_of_the_Jedi_(1983)', topk=100)]
        self.assertIn('229.Star_Trek_IV:_The_Voyage_Home_(1986)', ret_a)
        self.assertFalse(hasattr(c, 'P'))
        c.normalize(group='item')
        ret_a = [x for x, _ in c.most_similar('180.Return_of_the_Jedi_(1983)', topk=100)]
        self.assertIn('229.Star_Trek_IV:_The_Voyage_Home_(1986)', ret_a)

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
