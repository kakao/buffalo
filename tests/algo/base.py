# -*- coding: utf-8 -*-
import os
import unittest

from buffalo.misc import aux
from buffalo.misc.log import set_log_level
from buffalo.data.mm import MatrixMarketOptions


class TestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isdir('ml-100k/'):
            raise RuntimeError('Cannot find the ./ml-100k directory')
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
                    userid = line.strip().split('|')[0]
                    fout.write('%s\n' % userid)
        cls.ml_100k = './ml-100k/'
        if not os.path.isdir('ml-20m'):
            raise RuntimeError('Cannot find the ./ml-20m directory')

        if not os.path.isfile('./ml-20m/main'):
            uids, iids = {}, {}
            with open('./ml-20m/ratings.csv') as fin:
                fin.readline()
                for line in fin:
                    uid = line.split(',')[0]
                    if uid not in uids:
                        uids[uid] = len(uids) + 1
            with open('./ml-20m/uid', 'w') as fout:
                for uid, _ in sorted(uids.items(), key=lambda x: x[1]):
                    fout.write('%s\n' % uid)
            with open('./ml-20m/movies.csv') as fin:
                fin.readline()
                for line in fin:
                    iid = line.split(',')[0]
                    iids[iid] = len(iids) + 1
            with open('./ml-20m/iid', 'w') as fout:
                for iid, _ in sorted(iids.items(), key=lambda x: x[1]):
                    fout.write('%s\n' % iid)
            with open('./ml-20m/main', 'w') as fout:
                fout.write('%%MatrixMarket matrix coordinate real general\n%\n%\n138493 27278 20000263\n')
                with open('./ml-20m/ratings.csv') as fin:
                    fin.readline()
                    for line in fin:
                        uid, iid, r, *_ = line.split(',')
                        uid, iid = uids[uid], iids[iid]
                        fout.write(f'{uid} {iid} {r}\n')
        cls.ml_20m = './ml-20m/'
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
        self.assertTrue(results['ndcg'] > ndcg)
        self.assertTrue(results['map'] > map)

    def _test6_topk(self, cls, opt):
        set_log_level(1)

        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.ml_100k + 'main'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.value_prepro = aux.Option({'name': 'OneBased'})

        c = cls(opt, data_opt=data_opt)
        c.initialize()
        c.train()
        c.data.build_idmaps()
        self.assertTrue(len(c.topk_recommendation('1', 10)['1']), 10)
        c.fast_similar(True)
        ret_a = c.most_similar('Star_Wars_(1977)', 10)['Star_Wars_(1977)']
        self.assertIn('Return_of_the_Jedi_(1983)', ret_a)
        c.fast_similar(False)
        ret_b = c.most_similar('Star_Wars_(1977)', 10)['Star_Wars_(1977)']
        self.assertIn('Return_of_the_Jedi_(1983)', ret_b)
        self.assertEqual(ret_a, ret_b)

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
        c.data.build_idmaps()
        ret_a = c.most_similar('Star_Wars_(1977)', 10)['Star_Wars_(1977)']
        self.assertIn('Return_of_the_Jedi_(1983)', ret_a)
        c.save('model.bin')
        c.load('model.bin')
        os.remove('model.bin')
        ret_a = c.most_similar('Star_Wars_(1977)', 10)['Star_Wars_(1977)']
        self.assertIn('Return_of_the_Jedi_(1983)', ret_a)
