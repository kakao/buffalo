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
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('''%%MatrixMarket matrix coordinate integer general\n%\n%\n5 3 5\n1 1 1\n2 1 3\n3 3 1\n4 2 1\n5 2 2''')
            cls.mm_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('''lucas\ngony\nnagari\nkim\nlee''')
            cls.uid_path = f.name
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('''apple\nmango\nbanana''')
            cls.iid_path = f.name
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
        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.input.main = self.mm_path
        data_opt.input.uid = self.uid_path
        data_opt.input.iid = self.iid_path

        als = ALS(opt, data_opt=data_opt)
        als.init_factors()
        als.train()


if __name__ == '__main__':
    unittest.main()
