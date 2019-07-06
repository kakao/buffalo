# -*- coding: utf-8 -*-
import os
import unittest


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
                    fout.write('%s\n' % line.strip())
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
