# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import time
import unittest
from itertools import combinations

import numpy as np

from buffalo.algo.als import ALS
from buffalo.algo.w2v import W2V
from buffalo.algo.bpr import BPRMF
from buffalo.misc.log import set_log_level
from buffalo.data.stream import StreamOptions
from buffalo.data.mm import MatrixMarketOptions
from buffalo.parallel.base import ParALS, ParBPRMF, ParW2V
from buffalo.algo.options import ALSOption, BPRMFOption, W2VOption

from .base import TestBase


class TestAlgo(TestBase):
    def load_text8_model(self):
        if os.path.isfile('text8.w2v.bin'):
            w2v = W2V()
            w2v.load('text8.w2v.bin')
            return w2v
        set_log_level(3)
        opt = W2VOption().get_default_option()
        opt.num_workers = 12
        opt.d = 40
        opt.min_count = 4
        opt.num_iters = 10
        opt.model_path = 'text8.w2v.bin'
        data_opt = StreamOptions().get_default_option()
        data_opt.input.main = self.text8 + 'main'
        data_opt.data.path = './text8.h5py'
        data_opt.data.use_cache = True
        data_opt.data.validation = {}

        c = W2V(opt, data_opt=data_opt)
        c.initialize()
        c.train()
        c.save()
        return c

    def get_ml100k_mm_opt(self):
        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.use_cache = True
        data_opt.data.path = './ml100k.h5py'
        return data_opt

    def test01_most_similar(self):
        set_log_level(2)
        data_opt = self.get_ml100k_mm_opt()
        opt = ALSOption().get_default_option()
        opt.d = 20
        opt.num_workers = 1
        als = ALS(opt, data_opt=data_opt)
        als.initialize()
        als.train()
        pals = ParALS(als)
        random_keys = [k for k, _ in als.most_similar('49.Star_Wars_(1977)', topk=128)]
        random_indexes = als.get_index_pool(random_keys)
        naive = [als.most_similar(k, topk=10) for k in random_keys]
        topks0 = [[k for k, _ in result] for result in naive]
        scores0 = np.array([[v for _, v in result] for result in naive])
        self.assertEqual(scores0.shape, (128, 10,), msg='check even size')
        scores0 = scores0.reshape(len(naive), 10)
        pals.num_workers = 1
        topks1, scores1 = pals.most_similar(random_keys, topk=10, repr=True)
        topks2, scores2 = pals.most_similar(random_indexes, topk=10, repr=True)

        for a, b in combinations([topks0, topks1, topks2], 2):
            self.assertEqual(a, b)
        for a, b in combinations([scores0, scores1, scores2], 2):
            self.assertTrue(np.allclose(a, b, atol=1e-07))

    def test02_most_similar(self):
        set_log_level(1)
        data_opt = self.get_ml100k_mm_opt()
        opt = ALSOption().get_default_option()
        opt.d = 20
        opt.num_workers = 1
        als = ALS(opt, data_opt=data_opt)
        als.initialize()
        als.train()
        als.build_itemid_map()
        pals = ParALS(als)

        all_keys = als._idmanager.itemids[::]
        start_t = time.time()
        [als.most_similar(k, topk=10) for k in all_keys]
        naive_elapsed = time.time() - start_t

        pals.num_workers = 4
        start_t = time.time()
        pals.most_similar(all_keys, topk=10, repr=True)
        parals_elapsed = time.time() - start_t

        self.assertTrue(naive_elapsed > parals_elapsed * 3.0)

    def test03_most_similar(self):
        set_log_level(1)
        data_opt = self.get_ml100k_mm_opt()
        opt = BPRMFOption().get_default_option()
        opt.d = 20
        opt.num_workers = 1
        bpr = BPRMF(opt, data_opt=data_opt)
        bpr.initialize()
        bpr.train()
        bpr.build_itemid_map()
        parbpr = ParBPRMF(bpr)

        all_keys = bpr._idmanager.itemids[::]
        start_t = time.time()
        [bpr.most_similar(k, topk=10) for k in all_keys]
        naive_elapsed = time.time() - start_t

        parbpr.num_workers = 4
        start_t = time.time()
        parbpr.most_similar(all_keys, topk=10, repr=True)
        parbpr_elapsed = time.time() - start_t

        self.assertTrue(naive_elapsed > parbpr_elapsed * 3.0)

    def test04_text8_most_similar(self):
        set_log_level(1)
        model = self.load_text8_model()
        par = ParW2V(model)

        model.opt.num_workers = 1
        all_keys = model._idmanager.itemids[::][:10000]
        start_t = time.time()
        [model.most_similar(k, topk=10) for k in all_keys]
        naive_elapsed = time.time() - start_t

        par.num_workers = 4
        start_t = time.time()
        par.most_similar(all_keys, topk=10, repr=True)
        par_elapsed = time.time() - start_t

        self.assertTrue(naive_elapsed > par_elapsed * 3.0)

    def test05_topk_MT(self):
        set_log_level(2)
        data_opt = self.get_ml100k_mm_opt()
        opt = ALSOption().get_default_option()
        opt.d = 20
        opt.num_workers = 1
        als = ALS(opt, data_opt=data_opt)
        als.initialize()
        als.train()

        als.build_userid_map()
        all_keys = als._idmanager.userids
        start_t = time.time()
        naive = als.topk_recommendation(all_keys, topk=5)
        naive_elapsed = time.time() - start_t

        pals = ParALS(als)
        pals.num_workers = 4
        start_t = time.time()
        qkeys1, topks1, scores1 = pals.topk_recommendation(all_keys, topk=5, repr=True)
        par_elapsed = time.time() - start_t
        self.assertEqual(len(qkeys1), len(naive))
        for q, t in zip(qkeys1, topks1):
            self.assertEqual(naive[q], t)
        self.assertTrue(naive_elapsed > par_elapsed * 1.5)

    def test06_topk_pool(self):
        set_log_level(2)
        data_opt = self.get_ml100k_mm_opt()
        opt = ALSOption().get_default_option()
        opt.d = 20
        opt.num_workers = 1
        als = ALS(opt, data_opt=data_opt)
        als.initialize()
        als.train()
        pals = ParALS(als)

        pool = np.array([i for i in range(5)], dtype=np.int32)
        als.build_userid_map()
        all_keys = als._idmanager.userids[::][:10]
        naive = als.topk_recommendation(all_keys, topk=10, pool=pool)
        qkeys1, topks1, scores1 = pals.topk_recommendation(all_keys, topk=10, pool=pool, repr=True)
        for q, t in zip(qkeys1, topks1):
            self.assertEqual(naive[q], t)

    def test07_topk_pool(self):
        set_log_level(2)
        data_opt = self.get_ml100k_mm_opt()
        opt = BPRMFOption().get_default_option()
        opt.d = 20
        opt.num_workers = 1
        model = BPRMF(opt, data_opt=data_opt)
        model.initialize()
        model.train()
        par = ParBPRMF(model)

        pool = np.array([i for i in range(5)], dtype=np.int32)
        model.build_userid_map()
        all_keys = model._idmanager.userids[::][:10]
        naive = model.topk_recommendation(all_keys, topk=10, pool=pool)
        qkeys1, topks1, scores1 = par.topk_recommendation(all_keys, topk=10, pool=pool, repr=True)
        for q, t in zip(qkeys1, topks1):
            self.assertEqual(naive[q], t)


if __name__ == '__main__':
    unittest.main()
