# -*- coding: utf-8 -*-
import os

import unittest

from buffalo.algo.w2v import W2V
from buffalo.misc.log import set_log_level
from buffalo.algo.options import W2VOption
from buffalo.data.stream import StreamOptions

from .base import TestBase


class TestW2V(TestBase):
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

    def test00_get_default_option(self):
        W2VOption().get_default_option()
        self.assertTrue(True)

    def test01_is_valid_option(self):
        opt = W2VOption().get_default_option()
        self.assertTrue(W2VOption().is_valid_option(opt))
        opt['save_best'] = 1
        self.assertRaises(RuntimeError, W2VOption().is_valid_option, opt)
        opt['save_best'] = False
        self.assertTrue(W2VOption().is_valid_option(opt))

    def test02_init_with_dict(self):
        set_log_level(3)
        opt = W2VOption().get_default_option()
        W2V(opt)
        self.assertTrue(True)

    def test03_init(self):
        pass

    def test04_train(self):
        self.load_text8_model()
        self.assertTrue(True)

    def test05_text8_accuracy(self):
        set_log_level(2)
        opt = W2VOption().get_default_option()
        opt.num_workers = 12
        opt.d = 200
        opt.num_iters = 15
        opt.min_count = 4
        data_opt = StreamOptions().get_default_option()
        data_opt.input.main = self.text8 + 'main'
        data_opt.data.path = './text8.h5py'
        data_opt.data.use_cache = True
        data_opt.data.validation = {}

        model_path = 'text8.accuracy.w2v.bin'
        w = W2V(opt, data_opt=data_opt)
        if os.path.isfile(model_path):
            w.load(model_path)
        else:
            w.initialize()
            w.train()
            w.build_itemid_map()

        with open('./ext/text8/questions-words.txt') as fin:
            questions = fin.read().strip().split('\n')

        met = {}
        target_class = ['capital-common-countries']
        class_name = None
        for line in questions:
            if not line:
                continue
            if line.startswith(':'):
                _, class_name = line.split(' ', 1)
                if class_name in target_class and class_name not in met:
                    met[class_name] = {'hit': 0, 'miss': 0, 'total': 0}
            else:
                if class_name not in target_class:
                    continue
                a, b, c, answer = line.lower().strip().split()
                oov = any([w.get_feature(t) is None for t in [a, b, c, answer]])
                if oov:
                    continue
                topk = w.most_similar(w.get_weighted_feature({b: 1, c: 1, a: -1}))
                for nn, _ in topk:
                    if nn in [a, b, c]:
                        continue
                    if nn == answer:
                        met[class_name]['hit'] += 1
                    else:
                        met[class_name]['miss'] += 1
                    break  # top-1
                met[class_name]['total'] += 1
        stat = met['capital-common-countries']
        acc = float(stat['hit']) / stat['total']
        print('Top1-Accuracy={:0.3f}'.format(acc))
        self.assertTrue(acc > 0.7)

    def test06_most_similar(self):
        w = self.load_text8_model()
        q1, q2, q3 = 'apple', 'macintosh', 'microsoft'
        self._test_most_similar(w, q1, q2, q3)


if __name__ == '__main__':
    unittest.main()
