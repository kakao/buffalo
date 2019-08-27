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
        if not os.path.isdir('ext/ml-100k/'):
            raise RuntimeError('Cannot find the ./ml-100k directory')
        cls.ml_100k = './ext/ml-100k/'

        if not os.path.isdir('ext/ml-20m'):
            raise RuntimeError('Cannot find the ./ml-20m directory')
        cls.ml_20m = './ext/ml-20m/'

        if not os.path.isdir('ext/text8'):
            raise RuntimeError('Cannot find the ./text8 directory')
        cls.text8 = './ext/text8/'
        cls.temp_files = []

    @classmethod
    def tearDownClass(cls):
        for path in cls.temp_files:
            os.remove(path)
