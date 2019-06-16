# -*- coding: utf-8 -*-
from buffalo.misc.aux import InputOptions, Option


class AlsOption(InputOptions):
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
        return Option(opt)
