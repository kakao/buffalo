# -*- coding: utf-8 -*-
import unittest
from .base import TestBase
from buffalo.algo.cfr import CFR
from buffalo.misc.log import set_log_level
from buffalo.algo.options import CFROption
from buffalo.data.stream import StreamOptions


class TestCFR(TestBase):
    '''
    def test0_get_default_option(self):
        CFROption().get_default_option()
        self.assertTrue(True)

    def test1_is_valid_option(self):
        opt = CFROption().get_default_option()
        self.assertTrue(CFROption().is_valid_option(opt))
        opt['save_best'] = 1
        self.assertRaises(RuntimeError, CFROption().is_valid_option, opt)
        opt['save_best'] = False
        self.assertTrue(CFROption().is_valid_option(opt))

    def test2_init_with_dict(self):
        set_log_level(3)
        opt = CFROption().get_default_option()
        CFR(opt)
        self.assertTrue(True)
    '''

    def test3_init(self):
        opt = CFROption().get_default_option()
        opt.dim = 20
        set_log_level(3)
        data_opt = StreamOptions().get_default_option()
        data_opt.data.sppmi = {"enabled": True, "windows": 5, "k": 10}
        data_opt.data.internal_data_type = "matrix"
        data_opt.input.main = self.ml_100k + 'stream'
        data_opt.input.uid = self.ml_100k + 'uid'
        data_opt.input.iid = self.ml_100k + 'iid'
        data_opt.data.path = './ml100k.h5py'

        c = CFR(opt, data_opt=data_opt)
        self.assertTrue(True)
        c.initialize()
        self.assertEqual(c.U.shape, (943, 20))
        self.assertEqual(c.I.shape, (1682, 20))


if __name__ == '__main__':
    unittest.main()
