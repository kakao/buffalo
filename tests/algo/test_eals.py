import unittest

from loguru import logger

import buffalo
from buffalo import EALS, EALSOption, aux, set_log_level, MatrixMarketOptions, set_log_level

from .base import TestBase


class TestEALS(TestBase):
    def test00_get_default_option(self):
        EALSOption().get_default_option()
        self.assertTrue(True)

    def test01_is_valid_option(self):
        opt = EALSOption().get_default_option()
        self.assertTrue(EALSOption().is_valid_option(opt))
        opt["save_best"] = 1
        self.assertRaises(RuntimeError, EALSOption().is_valid_option, opt)
        opt["save_best"] = False
        self.assertTrue(EALSOption().is_valid_option(opt))

    def test02_init_with_dict(self):
        set_log_level(3)
        opt = EALSOption().get_default_option()
        EALS(opt)
        self.assertTrue(True)

    def test03_init(self):
        opt = EALSOption().get_default_option()
        self._test3_init(EALS, opt)

    def test04_train(self):
        opt = EALSOption().get_default_option()
        opt.d = 20
        self._test4_train(EALS, opt)

    def test05_validation(self):
        opt = EALSOption().get_default_option()
        opt.d = 5
        opt.num_iters = 20
        opt.validation = aux.Option({"topk": 10})
        self._test5_validation(EALS, opt)

    def test05_1_validation_with_callback(self,):
        opt = EALSOption().get_default_option()
        opt.d = 5
        opt.num_iters = 20
        opt.validation = aux.Option({"topk": 10})
        self._test5_1_validation_with_callback(EALS, opt)

    def test06_topk(self):
        opt = EALSOption().get_default_option()
        opt.d = 5
        opt.validation = aux.Option({"topk": 10})
        self._test6_topk(EALS, opt)

    def test07_train_ml_20m(self):
        opt = EALSOption().get_default_option()
        opt.num_workers = 8
        opt.validation = aux.Option({"topk": 10})
        self._test7_train_ml_20m(EALS, opt)
    
    def test08_serialization(self):
        opt = EALSOption().get_default_option()
        opt.d = 5
        opt.validation = aux.Option({"topk": 10})
        self._test8_serialization(EALS, opt)

    def test09_compact_serialization(self):
        opt = EALSOption().get_default_option()
        opt.d = 5
        opt.validation = aux.Option({"topk": 10})
        self._test9_compact_serialization(EALS, opt)

    def test10_fast_most_similar(self):
        opt = EALSOption().get_default_option()
        opt.d = 5
        opt.validation = aux.Option({"topk": 10})
        self._test10_fast_most_similar(EALS, opt)


if __name__ == "__main__":
    unittest.main()
