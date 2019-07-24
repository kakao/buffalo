# -*- coding: utf-8 -*-
from buffalo.misc.aux import InputOptions, Option


class AlgoOption(InputOptions):
    def __init__(self, *args, **kwargs):
        super(AlgoOption, self).__init__(*args, **kwargs)

    def get_default_option(self):
        opt = {
        }
        return opt

    def get_default_optimize_option(self):
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

            'model_path': '',
            'data_opt': {}
        })
        return Option(opt)

    def get_default_optimize_option(self):
        """Optimization Options for ALS
        options:
            loss(str): Target loss to optimize.
            max_trials(int, option): Maximum experiments for optimization. If not given, run forever.
            min_trials(int, option): Minimum experiments before deploying model. (Since the best parameter may not be found after `min_trials`, the first best parameter is always deployed)
            deployment(bool): Set True to train model with the best parameter. During the optimization, it try to dump the model which beated the previous best loss.
            start_with_default_parameters(bool): If set to True, the loss value of the default parameter is used as the starting loss to beat.
            space(dict): Parameter space definition. For more information, pleases reference hyperopt's express. Note) Due to hyperopt's `randint` does not provide lower value, we had to implement it a bait tricky. Pleases see optimize.py to check how we deal with `randint`.k
        """
        opt = super().get_default_optimize_option()
        opt.update({
            'loss': 'rmse',
            'max_trials': 100,
            'min_trials': 0,
            'deployment': True,
            'start_with_default_parameters': True,
            'space': {
                'adaptive_reg': ['choice', ['adaptive_reg', [0, 1]]],
                'd': ['randint', ['d', 10, 30]],
                'reg_u': ['uniform', ['reg_u', 0.1, 1]],
                'reg_i': ['uniform', ['reg_i', 0.1, 1]],
                'alpha': ['randint', ['alpha', 1, 32]]
            }
        })
        return Option(opt)
