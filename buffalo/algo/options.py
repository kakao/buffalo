from buffalo.misc.aux import InputOptions, Option


class AlgoOption(InputOptions):
    def __init__(self, *args, **kwargs):
        super(AlgoOption, self).__init__(*args, **kwargs)

    def get_default_option(self):
        """Default options for Algo classes.

        :ivar bool evaluation_on_learning: Set True to do run evaluation on training phrase. (default: True)
        :ivar bool compute_loss_on_training: Set True to calculate loss on training phrase. (default: True)
        :ivar int early_stopping_rounds: The number of exceed epochs after reached minimum loss on training phrase. If set 0, it doesn't work. (default: 0)
        :ivar bool save_best: Whenever the loss improved, save the model.
        :ivar int evaluation_period: How often will do evaluation in epochs. (default: 1)
        :ivar int save_period: How often will do save_best routine in epochs. (default: 10)
        :ivar int random_seed: Random Seed
        :ivar dict validation: The validation options.
        """
        opt = {
            "evaluation_on_learning": True,
            "compute_loss_on_training": True,
            "early_stopping_rounds": 0,
            "save_best": False,
            "evaluation_period": 1,
            "save_period": 10,
            "random_seed": 0,
            "validation": {}
        }
        return opt

    def is_valid_option(self, opt):
        b = super().is_valid_option(opt)
        for f in ["num_workers"]:
            if f not in opt:
                raise RuntimeError(f"{f} not defined")
        return b


class ALSOption(AlgoOption):
    def __init__(self, *args, **kwargs):
        super(ALSOption, self).__init__(*args, **kwargs)

    def get_default_option(self):
        """Options for Alternating Least Squares.

        :ivar bool adaptive_reg: Set True, for adaptive regularization. (default: False)
        :ivar bool save_factors: Set True, to save models. (default: False)
        :ivar bool accelerator: Set True, to accelerate training using GPU. (default: False)
        :ivar int d: The number of latent feature dimension. (default: 20)
        :ivar int num_iters: The number of iterations for training. (default: 10)
        :ivar int num_workers: The number of threads. (default: 1)
        :ivar int hyper_threads: The number of hyper threads when using cuda cores. (default: 256)
        :ivar float reg_u: The L2 regularization coefficient for user embedding matrix. (default: 0.1)
        :ivar float reg_i: The L2 regularization coefficient for item embedding matrix. (default: 0.1)
        :ivar float alpha: The coefficient of giving more weights to losses on positive samples. (default: 8)
        :ivar float eps: epsilon for numerical stability (default: 1e-10)
        :ivar float cg_tolerance: tolerance of conjugate gradient for early stopping iterations (default: 1e-10)
        :ivar int block_size: block size for ialspp optimizer. Only enabled with "ialspp" optimizer.
        :ivar str optimizer: The name of optimizer, should be in [llt, ldlt, manual_cg, eigen_cg, eigen_bicg, eigen_gmres, eigen_dgmres, eigen_minres, ialspp]. (default: manual_cg)
        :ivar int num_cg_max_iters: The number of maximum iterations for conjugate gradient optimizer. (default: 3)
        :ivar str model_path: Where to save model.
        :ivar dict data_opt: This option will be used to load data if given.
        """

        opt = super().get_default_option()
        opt.update({
            "adaptive_reg": False,
            "save_factors": False,
            "accelerator": False,
            "d": 20,
            "num_iters": 10,
            "num_workers": 1,
            "hyper_threads": 256,
            "num_cg_max_iters": 3,
            "reg_u": 0.1,
            "reg_i": 0.1,
            "alpha": 8,
            "optimizer": "manual_cg",
            "cg_tolerance": 1e-10,
            "block_size": 32,
            "eps": 1e-10,
            "model_path": "",
            "data_opt": {}
        })
        return Option(opt)

    def is_valid_option(self, opt):
        b = super().is_valid_option(opt)
        possible_optimizers = ["llt", "ldlt", "manual_cg", "eigen_cg", "eigen_bicg",
                               "eigen_gmres", "eigen_dgmres", "eigen_minres", "ialspp"]
        if opt.optimizer not in possible_optimizers:
            msg = f"optimizer ({opt.optimizer}) should be in {possible_optimizers}"
            raise RuntimeError(msg)
        return b


class CFROption(AlgoOption):
    def __init__(self, *args, **kwargs):
        super(CFROption, self).__init__(*args, **kwargs)

    def get_default_option(self):
        """ Basic Options for CoFactor.

        :ivar int d: The number of latent feature dimension. (default: 20)
        :ivar int num_iters: The number of iterations for training. (default: 10)
        :ivar int num_workers: The number of threads. (default: 1)
        :ivar float reg_u: The L2 regularization coefficient for user embedding matrix. (default: 0.1)
        :ivar float reg_i: The L2 regularization coefficient for item embedding matrix. (default: 0.1)
        :ivar float reg_c: The L2 regularization coefficient for context embedding matrix. (default: 0.1)
        :ivar float eps: epsilon for numerical stability (default: 1e-10)
        :ivar float cg_tolerance: The tolerance for early stopping conjugate gradient optimizer. (default: 1e-10)
        :ivar float alpha: The coefficient of giving more weights to losses on positive samples. (default: 8.0)
        :ivar float l: The relative weight of loss on user-item relation over item-context relation. (default: 1.0)
        :ivar str optimizer: The name of optimizer, should be in [llt, ldlt, manual_cg, eigen_cg, eigen_bicg, eigen_gmres, eigen_dgmres, eigen_minres]. (default: manual_cg)
        :ivar int num_cg_max_iters: The number of maximum iterations for conjugate gradient optimizer. (default: 3)
        :ivar str model_path: Where to save model. (default: "")
        :ivar dict data_opt: This option will be used to load data if given. (default: {})
        """
        opt = super().get_default_option()
        opt.update({
            "save_factors": False,
            "d": 20,
            "num_iters": 10,
            "num_workers": 1,
            "num_cg_max_iters": 3,

            "cg_tolerance": 1e-10,
            "eps": 1e-10,
            "reg_u": 0.1,
            "reg_i": 0.1,
            "reg_c": 0.1,
            "alpha": 8.0,
            "l": 1.0,

            "optimizer": "manual_cg",
            "model_path": "",
            "data_opt": {}
        })
        return Option(opt)

    def is_valid_option(self, opt):
        b = super().is_valid_option(opt)
        possible_optimizers = ["llt", "ldlt", "manual_cg", "eigen_cg", "eigen_bicg",
                               "eigen_gmres", "eigen_dgmres", "eigen_minres"]
        if opt.optimizer not in possible_optimizers:
            msg = f"optimizer ({opt.optimizer}) should be in {possible_optimizers}"
            raise RuntimeError(msg)
        return b


class BPRMFOption(AlgoOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_default_option(self):
        """Options for Bayesian Personalized Ranking Matrix Factorization.

        :ivar bool accelerator: Set True, to accelerate training using GPU. (default: False)
        :ivar bool use_bias: Set True, to use bias term for the model.
        :ivar int evaluation_period: (default: 100)
        :ivar int num_workers: The number of threads. (default: 1)
        :ivar int hyper_threads: The number of hyper threads when using cuda cores. (default: 256)
        :ivar int num_iters: The number of iterations for training. (default: 100)
        :ivar int d: The number of latent feature dimension. (default: 20)
        :ivar bool update_i: Set True, to update positive item feature. (default: True)
        :ivar bool update_j: Set True, to update negative item feature. (default: True)
        :ivar float reg_u: The L2 regularization coefficient for user embedding matrix. (default: 0.025)
        :ivar float reg_i: The L2 regularization coefficient for positive item embedding matrix. (default: 0.025)
        :ivar float reg_j: The L2 regularization coefficient for negative item embedding matrix. (default: 0.025)
        :ivar float reg_b: The L2 regularization coefficient for bias term. (default: 0.025)
        :ivar str optimizer: The name of optimizer, should be one of [sgd, adagrad, adam]. (default: sgd)
        :ivar float lr: The learning rate.
        :ivar float min_lr: The minimum of learning rate, to prevent going to zero by learning rate decaying. (default: 0.0001)
        :ivar float beta1: The parameter of Adam optimizer. (default: 0.9)
        :ivar float beta2: The parameter of Adam optimizer. (default: 0.999)
        :ivar bool per_coordinate_normalize: This is a bit tricky option for Adam optimizer. Before update factors with gradients, do normalize gradients per class by its number of contributed samples. (default: False)
        :ivar float sampling_power: This parameter control sampling distribution. When it set to 0, it draw negative items from uniform distribution, while to set 1, it draw from the given data popularation. (default: 0.0)
        :ivar bool random_positive: Set True, to draw positive sample uniformly instead of using straight forward positive sample, only implemented in cuda mode, according to the original paper, set True, but we found out False usually produces better results) (default: False)
        :ivar bool verify_neg: Set True, to ensure negative sample does not belong to positive samples. (default True)
        :ivar str model_path: Where to save model.
        :ivar dict data_opt: This option will be used to load data if given.
        """
        opt = super().get_default_option()
        opt.update({
            "accelerator": False,
            "use_bias": True,
            "evaluation_period": 100,
            "num_workers": 1,
            "hyper_threads": 256,
            "num_iters": 100,
            "d": 20,
            "update_i": True,
            "update_j": True,
            "reg_u": 0.025,
            "reg_i": 0.025,
            "reg_j": 0.025,
            "reg_b": 0.025,

            "optimizer": "sgd",
            "lr": 0.002,
            "min_lr": 0.0001,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-10,

            "per_coordinate_normalize": False,
            "num_negative_samples": 1,
            "sampling_power": 0.0,
            "verify_neg": True,
            "random_positive": False,

            "model_path": "",
            "data_opt": {}
        })
        return Option(opt)


class WARPOption(AlgoOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_default_option(self):
        """Options for WARP Matrix Factorization.

        :ivar bool accelerator: Set True, to accelerate training using GPU. (default: False)
        :ivar int evaluation_period: (default: 15)
        :ivar int num_workers: The number of threads. (default: 1)
        :ivar int hyper_threads: The number of hyper threads when using cuda cores. (default: 256)
        :ivar int num_iters: The number of iterations for training. (default: 15)
        :ivar int d: The number of latent feature dimension. (default: 30)
        :ivar int max_trials: The maximum number of attempts to find a violating negative sample during training.
        :ivar str score_func: score function to use: either ["dot", "l2"]
        :ivar bool update_i: Set True, to update positive item feature. (default: True)
        :ivar bool update_j: Set True, to update negative item feature. (default: True)
        :ivar float reg_u: The L2 regularization coefficient for user embedding matrix. (default: 0.0)
        :ivar float reg_i: The L2 regularization coefficient for positive item embedding matrix. (default: 0.0)
        :ivar float reg_j: The L2 regularization coefficient for negative item embedding matrix. (default: 0.0)
        :ivar str optimizer: The name of optimizer, should be one of [adagrad, adam]. (default: adagrad)
        :ivar float lr: The learning rate. (default: 0.1)
        :ivar float min_lr: The minimum of learning rate, to prevent going to zero by learning rate decaying. (default: 0.0001)
        :ivar float beta1: The parameter of Adam optimizer. (default: 0.9)
        :ivar float beta2: The parameter of Adam optimizer. (default: 0.999)
        :ivar bool per_coordinate_normalize: This is a bit tricky option for Adam optimizer. Before update factors with gradients, do normalize gradients per class by its number of contributed samples. (default: False)
        :ivar bool random_positive: Set True, to draw positive sample uniformly instead of using straight forward positive sample, only implemented in cuda mode, according to the original paper, set True, but we found out False usually produces better results) (default: False)
        :ivar str model_path: Where to save model.
        :ivar dict data_opt: This option will be used to load data if given.
        """
        opt = super().get_default_option()
        opt.update({
            "accelerator": False,
            "evaluation_period": 5,
            "num_workers": 1,
            "hyper_threads": 256,
            "num_iters": 40,
            "d": 64,
            "threshold": 1.0,
            "score_func": "dot",
            "max_trials": 500,
            "update_i": True,
            "update_j": True,
            "reg_u": 0.0,
            "reg_i": 0.0,
            "reg_j": 0.0,
            "optimizer": "adagrad",
            "lr": 0.05,
            "min_lr": 0.0001,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-10,
            "per_coordinate_normalize": False,
            "model_path": "",
            "data_opt": {}
        })
        return Option(opt)


class W2VOption(AlgoOption):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_default_option(self):
        """Options for Word2Vec.

        :ivar bool evaluation_on_learning: Set True to do run evaluation on training phrase. (default: False)
        :ivar int num_workers: The number of threads. (default: 1)
        :ivar int num_iters: The number of iterations for training. (default: 100)
        :ivar int d: The number of latent feature dimension. (default: 20)
        :ivar int window: The window size. (default: 5)
        :ivar int min_count: The minimum required frequency of the words to use training vocabulary. (default: 5)
        :ivar float sample: The sampling ratio to downsample the frequent words. (default: 0.001)
        :ivar int num_negative_samples: The number of negative noise words. (default: 5)
        :ivar float lr: The learning rate.
        :ivar str model_path: Where to save model.
        :ivar dict data_opt: This option will be used to load data if given.
        """
        opt = super().get_default_option()
        opt.update({
            "evaluation_on_learning": False,

            "num_workers": 1,
            "num_iters": 3,
            "d": 20,
            "window": 5,
            "min_count": 5,
            "sample": 0.001,
            "num_negative_samples": 5,

            "lr": 0.025,
            "min_lr": 0.0001,

            "model_path": "",
            "data_opt": {}
        })
        return Option(opt)


class PLSIOption(AlgoOption):
    def __init__(self, *args, **kwargs):
        super(PLSIOption, self).__init__(*args, **kwargs)

    def get_default_option(self):
        """ Basic Options for pLSI.

        :ivar int d: The number of latent feature dimension. (default: 20)
        :ivar int num_iters: The number of iterations for training. (default: 10)
        :ivar int num_workers: The number of threads. (default: 1)
        :ivar float alpha1: The coefficient of regularization term for clustering assignment. (default: 1.0)
        :ivar float alpha2: The coefficient of regularization term for item preference in each cluster. (default: 1.0)
        :ivar float eps: epsilon for numerical stability (default: 1e-10)
        :ivar str model_path: Where to save model. (default: "")
        :ivar bool save_factors: Set True, to save models. (default: False)
        :ivar dict data_opt: This option will be used to load data if given. (default: {})
        """
        opt = super().get_default_option()
        opt.update({
            "d": 20,
            "num_iters": 10,
            "num_workers": 1,
            "alpha1": 1.0,
            "alpha2": 1.0,
            "eps": 1e-10,
            "model_path": "",
            "save_factors": False,
            "data_opt": {},
            "inherit_opt": {}
        })
        return Option(opt)
