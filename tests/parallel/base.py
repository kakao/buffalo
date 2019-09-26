# -*- coding: utf-8 -*-
import os
import unittest

from buffalo.parallel.base import Parallel


class MockParallel(Parallel):
    def __init__(self, algo, num_workers=1):
        super().__init__(algo, num_workers=num_workers)

    def most_similar(self, keys, topk=10, group='item', pool=None):
        pass

    def topk_recommendation(self, keys, topk=10, pool=None):
        pass


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
