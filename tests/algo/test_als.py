# -*- coding: utf-8 -*-
import os
import unittest
import tempfile
from buffalo.algo.als import ALS
from buffalo.algo.options import AlsOption
from buffalo.misc.log import set_log_level
from buffalo.data import MatrixMarketOptions


class TestALS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.isfile('ml-100k/'):
            raise RuntimeError('Cannot find the resource  on ./ml-100k directory, checkout: https://grouplens.org/datasets/movielens/100k/')
        if not os.path.isfile('./ml-100k/main'):
            with open('./ml-100k/main', 'w') as fout:
                fout.write('%%MatrixMarket matrix coordinate integer general\n%\n%\n944 1682 80000\n')
                with open('./ml-100k/u1.base') as fin:
                    for line in fin:
                        u, i, v, ts = line.strip().split('\t')
                        fout.write('%s %s %s\n' % (u, i, v))

            with open('./ml-100k/iid', 'w') as fout:
                with open('./ml-100k/u.item', encoding='ISO-8859-1') as fin:
                    for line in fin:
                        fout.write('%s\n' % line.strip().split('|')[1].encode('utf8'))

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

    def test0_get_default_option(self):
        AlsOption().get_default_option()
        self.assertTrue(True)

    def test0_is_valid_option(self):
        opt = AlsOption().get_default_option()
        self.assertTrue(AlsOption().is_valid_option(opt))
        opt['save_best_only'] = 1
        self.assertRaises(RuntimeError, AlsOption().is_valid_option, opt)
        opt['save_best_only'] = False
        self.assertTrue(AlsOption().is_valid_option(opt))

    def test1_init_with_dict(self):
        opt = AlsOption().get_default_option()
        ALS(opt)
        self.assertTrue(True)

    def test2_init(self):
        set_log_level(2)
        opt = AlsOption().get_default_option()
        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.mm_path
        data_opt.input.uid = self.uid_path
        data_opt.input.iid = self.iid_path

        als = ALS(opt, data_opt=data_opt)
        self.assertTrue(True)
        self.temp_files.append(data_opt.data.path)
        als.init_factors()
        self.assertTrue(als.P.shape, (5, 20))
        self.assertTrue(als.Q.shape, (3, 20))

    def test3_train(self):
        set_log_level(2)
        opt = AlsOption().get_default_option()
        opt.d = 5

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.mm_path
        data_opt.input.uid = self.uid_path
        data_opt.input.iid = self.iid_path

        als = ALS(opt, data_opt=data_opt)
        als.init_factors()
        als.train()


if __name__ == '__main__':
    unittest.main()
