# -*- coding: utf-8 -*-
import os
import unittest


class TestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.isfile('ml-100k/'):
            raise RuntimeError('Cannot find the resource  on ./ml-100k directory, checkout: https://grouplens.org/datasets/movielens/100k/')
        if not os.path.isfile('./ml-100k/main'):
            with open('./ml-100k/main', 'w') as fout:
                fout.write('%%MatrixMarket matrix coordinate integer general\n%\n%\n943 1682 80000\n')
                with open('./ml-100k/u1.base') as fin:
                    for line in fin:
                        u, i, v, ts = line.strip().split('\t')
                        fout.write('%s %s %s\n' % (u, i, v))

            with open('./ml-100k/iid', 'w') as fout:
                with open('./ml-100k/u.item', encoding='ISO-8859-1') as fin:
                    for line in fin:
                        fout.write('%s\n' % line.strip().split('|')[1].replace(' ', '_'))

            with open('./ml-100k/uid', 'w') as fout:
                for line in open('./ml-100k/u.user'):
                    fout.write('%s\n' % line.strip())
        cls.mm_path = './ml-100k/main'
        cls.iid_path = './ml-100k/iid'
        cls.uid_path = './ml-100k/uid'
        cls.temp_files = []

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.mm_path)
        os.remove(cls.uid_path)
        os.remove(cls.iid_path)
        for path in cls.temp_files:
            os.remove(path)
