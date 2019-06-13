# -*- coding: utf-8 -*-
import unittest
from buffalo.algo.als import ALS
from buffalo.algo.options import AlsOption


class TestALS(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
