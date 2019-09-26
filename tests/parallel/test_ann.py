# -*- coding: utf-8 -*-
import os
import time
import unittest

from n2 import HnswIndex

from buffalo.algo.w2v import W2V
from buffalo.parallel.base import ParW2V
from buffalo.algo.options import W2VOption
from buffalo.misc.log import set_log_level
from buffalo.data.stream import StreamOptions

from .base import TestBase


class TestAnn(TestBase):
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

    def test01_text8_most_similar(self):
        set_log_level(1)
        model = self.load_text8_model()
        index = HnswIndex(model.L0.shape[1])
        model.normalize('item')
        for f in model.L0:
            index.add_data(f)
        index.build(n_threads=4)
        index.save('n2.bin')

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

        start_t = time.time()
        par.set_hnsw_index('n2.bin', 'item')
        par.most_similar(all_keys, topk=10, repr=True)
        ann_elapsed = time.time() - start_t
        self.assertTrue(naive_elapsed > par_elapsed * 1.5 > ann_elapsed * 5.0,
                        msg=f'{naive_elapsed} > {par_elapsed} > {ann_elapsed}')
        index.unload()
        os.remove('n2.bin')


if __name__ == '__main__':
    unittest.main()
