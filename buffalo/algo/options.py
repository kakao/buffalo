# -*- coding: utf-8 -*-
from buffalo.misc.aux import InputOptions, Option


class AlgoOption(InputOptions):
    def __init__(self, *args, **kwargs):
        super(AlgoOption, self).__init__(*args, **kwargs)

    def get_default_option(self):
        opt = {
            'evaluation_on_learning': True,
            'compute_loss_on_training': True,
            'early_stopping_rounds': 0,
            'save_best': False,
            'evaluation_period': 1,
            'save_period': 10,

            'random_seed': 0,
        }
        return opt

    def get_default_optimize_option(self):
        opt = {
            'loss': 'train_loss',
        }
        return opt

    def get_default_tensorboard_option(self):
        opt = {
            'name': 'default',
            'root': './tb',
            'name_template': '{name}.{dtm}'
        }
        return opt

    def is_valid_option(self, opt):
        b = super().is_valid_option(opt)
        for f in ['num_workers']:
            if not f in opt:
                raise RuntimeError(f'{f} not defined')
        return b


class AlsOption(AlgoOption):
    def __init__(self, *args, **kwargs):
        super(AlsOption, self).__init__(*args, **kwargs)

    def get_default_option(self):
        opt = super().get_default_option()
        opt.update({
            'adaptive_reg': False,
            'save_factors': False,

            'd': 20,
            'num_iters': 10,
            'num_workers': 1,
            'reg_u': 0.1,
            'reg_i': 0.1,
            'alpha': 8,
            'optimizer': 'manual_cg',
            'num_cg_max_iters': 3,

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
            'loss': 'train_loss',
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

class CFROption(AlgoOption):
    def __init__(self, *args, **kwargs):
        super(CFROption, self).__init__(*args, **kwargs)

    def get_default_option(self):
        """ Basic Options for CoFactor
        options:
            dim(int): latent space dimension
            num_iters(int): number of iterations for training
            num_workers(int): number of threads
            num_cg_max_iters(int): number of maximum iterations for conjuaget gradient optimizer
            reg_u(float): L2 regularization coefficient for user embedding matrix
            reg_i(float): L2 regularization coefficient for item embedding matrix
            reg_c(float): L2 regularization coefficient for context embedding matrix
            cg_tolerance(float): tolerance for early stopping conjugate gradient optimizer
            alpha(float): coefficient of giving more weights to losses on positive samples
            l(float): relative weight of loss on user-item relation over item-context relation
            compute_loss(bool): true if one wants to compute train loss
            optimizer(string): optimizer, should be in [llt, ldlt, manual_cg, eigen_cg]
        """
        opt = super().get_default_option()
        opt.update({
            'save_factors': False,
            'dim': 20,
            'num_iters': 10,
            'num_workers': 1,
            'compute_loss': True,
            'cg_tolerance': 1e-10,
            'reg_u': 0.1,
            'reg_i': 0.1,
            'reg_c': 0.1,
            'alpha': 8.0,
            'l': 1.0,
            'optimizer': 'manual_cg',
            'num_cg_max_iters': 3,
            'model_path': '',
            'data_opt': {}
        })
        return Option(opt)

    def get_default_optimize_option(self):
        """Optimization Options for CoFactor
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
            'loss': 'train_loss',
            'max_trials': 100,
            'min_trials': 0,
            'deployment': True,
            'start_with_default_parameters': True,
            'space': {
                'd': ['randint', ['d', 10, 30]],
                'reg_u': ['uniform', ['reg_u', 0.1, 1]],
                'reg_i': ['uniform', ['reg_i', 0.1, 1]],
                'reg_c': ['uniform', ['reg_i', 0.1, 1]],
                'alpha': ['randint', ['alpha', 1, 32]],
                'l': ['randint', ['alpha', 1, 32]]
            }
        })
        return Option(opt)

    def is_valid_option(self, opt):
        b = super().is_valid_option(opt)
        possible_optimizers = ["llt", "ldlt", "manual_cg", "eigen_cg", "eigen_bicg",
                               "eigen_gmres", "eigen_dgmres", "eigen_minres"]
        if not opt.optimizer in possible_optimizers:
            msg = f"optimizer ({opt.optimizer}) should be in {possible_optimizers}"
            raise RuntimeError(msg)
        return b


class BprmfOption(AlgoOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_default_option(self):
        opt = super().get_default_option()
        opt.update({
            'use_bias': True,
            'evaluation_period': 100,
            'num_workers': 1,
            'num_iters': 100,
            'd': 20,
            'update_i': True,
            'update_j': True,
            'reg_u': 0.025,
            'reg_i': 0.025,
            'reg_j': 0.025,
            'reg_b': 0.025,

            'optimizer': 'sgd',
            'lr': 0.002,
            'lr_decay': 0.0,
            'min_lr': 0.0001,
            'beta1': 0.9,
            'beta2': 0.999,
            'batch_size': -1,

            'per_coordinate_normalize': False,
            'num_negative_samples': 1,
            'sampling_power': 0.0,

            'model_path': '',
            'data_opt': {}
        })
        return Option(opt)

    def get_default_optimize_option(self):
        """Optimization Options for BPRMF
        """
        opt = super().get_default_optimize_option()
        opt.update({
            'loss': 'train_loss',
            'max_trials': 100,
            'min_trials': 0,
            'deployment': True,
            'start_with_default_parameters': True,
            'space': {
                'd': ['randint', ['d', 10, 30]],
                'reg_u': ['uniform', ['reg_u', 0.1, 1]],
                'reg_i': ['uniform', ['reg_i', 0.1, 1]]
            }
        })
        return Option(opt)


class W2vOption(AlgoOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_default_option(self):
        opt = super().get_default_option()
        opt.update({
            'evaluation_on_learning': False,

            'num_workers': 1,
            'num_iters': 3,
            'd': 20,
            'window': 5,
            'min_count': 5,
            'sample': 0.001,

            'lr': 0.025,
            'min_lr': 0.0001,
            'batch_size': -1,

            'num_negative_samples': 5,

            'model_path': '',
            'data_opt': {}
        })
        return Option(opt)

    def get_default_optimize_option(self):
        """Optimization Options for W2V
        """
        #
        opt = super().get_default_optimize_option()
        opt.update({
            'loss': 'train_loss',
            'max_trials': 100,
            'min_trials': 0,
            'deployment': True,
            'start_with_default_parameters': True,
            'space': {
                'd': ['randint', ['d', 10, 30]],
                'window': ['randint', ['window', 2, 8]],
                'num_negative_samples': ['randint', ['alpha', 1, 12]]
            }
        })
        return Option(opt)
