# -*- coding: utf-8 -*-
import time
import psutil
import unittest

import numpy as np

from buffalo.algo.als import ALS
from buffalo.misc.log import set_log_level

from .base import TestBase, MockParallel


class TestParallelBase(TestBase):
    def get_factors(self, R, C):
        F = np.random.rand(R, C).astype(np.float32)
        F = F / np.sqrt((F ** 2).sum(-1) + 1e-8)[..., np.newaxis]
        return F

    def get_most_similar(self, indexes, F, topk):
        topk += 1
        scores = F[indexes].dot(F.T)
        topks = np.argsort(scores, axis=1)[:, -topk:][:, ::-1]
        topks = np.array([t[1:] for t in topks])
        scores = np.array([s[t] for t, s in zip(topks, scores)])
        return topks, scores

    def get_topk(self, indexes, P, F, topk):
        scores = P[indexes].dot(F.T)
        topks = np.argsort(scores, axis=1)[:, -topk:][:, ::-1]
        scores = np.array([s[t] for t, s in zip(topks, scores)])
        return topks, scores

    def test00_init(self):
        set_log_level(1)
        als = ALS()
        MockParallel(als)
        self.assertTrue(True)

    def test01_most_similar(self):
        set_log_level(1)
        als = ALS()
        mp = MockParallel(als)
        Q = self.get_factors(128, 5)
        indexes = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        pool = np.array([], dtype=np.int32)
        topks1, scores1 = mp._most_similar('item', indexes, Q, 10, pool, -1, True)
        topks2, scores2 = self.get_most_similar(indexes, Q, 10)
        self.assertTrue(np.allclose(topks1, topks2, atol=1e-07))
        self.assertTrue(np.allclose(scores1, scores2, atol=1e-07))

    def test02_most_similar(self):
        num_cpu = psutil.cpu_count()
        if num_cpu < 2:
            return
        set_log_level(1)
        als = ALS()
        mp = MockParallel(als)
        R = 1000000
        Q = self.get_factors(1000000, 12)
        indexes = np.random.choice(range(R), 1024).astype(np.int32)
        pool = np.array([], dtype=np.int32)
        elapsed = []
        results = []
        for num_workers in [1] + [i * 2
                                  for i in range(1, num_cpu + 1)
                                  if i * 2 < num_cpu][:3]:
            mp.num_workers = num_workers
            start_t = time.time()
            ret = mp._most_similar('item', indexes, Q, 10, pool, -1, True)
            elapsed.append(time.time() - start_t)
            results.append(ret)
        for i in range(1, len(elapsed)):
            self.assertTrue(elapsed[i - 1] > elapsed[i] * 1.2)
            self.assertTrue(np.allclose(results[i - 1][0], results[i][0], atol=1e-07))
            self.assertTrue(np.allclose(results[i - 1][1], results[i][1], atol=1e-07))

    def test03_pool(self):
        set_log_level(1)
        als = ALS()
        mp = MockParallel(als)
        Q = self.get_factors(128, 5)
        indexes = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        pool = np.array([5, 6, 7], dtype=np.int32)
        topks, scores = mp._most_similar('item', indexes, Q, 10, pool, -1, True)
        self.assertTrue(set(topks[::].reshape(10 * 5)), set([5, 6, 7, -1]))

    def test04_topk(self):
        set_log_level(1)
        als = ALS()
        mp = MockParallel(als)
        P = self.get_factors(512, 5)
        Q = self.get_factors(128, 5)
        q_indexes = np.array([312, 313, 314, 315, 316], dtype=np.int32)
        pool = np.array([], dtype=np.int32)
        topks1, scores1 = mp._topk_recommendation(q_indexes, P, Q, 10, pool)
        topks2, scores2 = self.get_topk(q_indexes, P, Q, 10)
        self.assertTrue(np.allclose(topks1, topks2, atol=1e-07))
        self.assertTrue(np.allclose(scores1, scores2, atol=1e-07))


if __name__ == '__main__':
    unittest.main()
