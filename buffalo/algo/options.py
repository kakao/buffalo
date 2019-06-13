# -*- coding: utf-8 -*-
from buffalo.algo.base import AlgoOption


class AlsOption(AlgoOption):
    def get_default_option(self):
        opt = {
            'adaptive_reg': False,
            'evaluation_on_learning': True,
            'save_factors': False,
            'save_best_only': False,

            'd': 20,
            'num_iters': 10,
            'num_workers': 1,
            'early_stopping_rounds': 5,
            'reg_u': 0.1,
            'reg_i': 0.1,
            'alpha': 8,
            'validation_p': 0.1,

            'user_factor_path': '',
            'item_factor_path': ''
        }
        return opt

    def is_valid_option(self, opt):
        default_opt = self.get_default_option()
        keys = self.get_default_option()
        for key in keys:
            if key not in opt:
                raise RuntimeError('{} not exists on Option'.format(key))
            if type(opt.get(key)) != type(default_opt[key]):
                raise RuntimeError('Invalid type for {}, {} expected. '.format(key, type(default_opt[key])))
        return True
