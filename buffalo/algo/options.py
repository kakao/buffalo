# -*- coding: utf-8 -*-
from buffalo.misc.aux import InputOptions, Option

class AlgoOption(InputOptions):
    def __init__(self, *args, **kwargs):
        super(AlgoOption, self).__init__(*args, **kwargs)

    def get_default_option(self):
        opt = {
        }
        return opt


class AlsOption(AlgoOption):
    def __init__(self, *args, **kwargs):
        super(AlsOption, self).__init__(*args, **kwargs)

    def get_default_option(self):
        opt = super().get_default_option()
        opt.update({
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

            'user_factor_path': '',
            'item_factor_path': ''
        })
        return Option(opt)
