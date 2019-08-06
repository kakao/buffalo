# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import time
import numpy as np
from implicit.datasets.movielens import get_movielens


class Benchmark(object):
    def get_option(self, lib_name, algo_name, **kwargs):
        if lib_name == 'buffalo':
            if algo_name == 'als':
                from buffalo.algo.options import AlsOption
                opt = AlsOption().get_default_option()
                opt.update({'d': kwargs.get('d', 100),
                            'use_conjugate_gradient': kwargs.get('use_cg', True),
                            'num_iters': kwargs.get('num_iters', 10),
                            'num_workers': kwargs.get('num_workers', 10),
                            'compute_loss_on_training': kwargs.get('compute_loss_on_training', False)})
                return opt
            if algo_name == 'bpr':
                from buffalo.algo.options import BprmfOption
                opt = BprmfOption().get_default_option()
                opt.update({'d': kwargs.get('d', 100),
                            'num_iters': kwargs.get('num_iters', 10),
                            'num_workers': kwargs.get('num_workers', 10),
                            'compute_loss_on_training': kwargs.get('compute_loss_on_training', False)})
                return opt
        elif lib_name == 'implicit':
            if algo_name == 'als':
                return {'factors': kwargs.get('d', 100),
                        'dtype': np.float32,
                        'use_native': True,
                        'use_cg': kwargs.get('use_cg', True),
                        'iterations': kwargs.get('num_iters', 10),
                        'num_threads': kwargs.get('num_workers', 10),
                        'calculate_training_loss': kwargs.get('calculate_training_loss', False)}
            if algo_name == 'bpr':
                return {'factors': kwargs.get('d', 100),
                        'dtype': np.float32,
                        'iterations': kwargs.get('num_iters', 10),
                        'verify_negative_samples': True,
                        'num_threads': kwargs.get('num_workers', 10)}


class ImplicitLib(Benchmark):
    def __init__(self):
        super().__init__()

    def get_database(self, name):
        if name == 'ml20m':
            _, ratings = get_movielens('20m')
            ratings.data = np.ones(len(ratings.data))
            ratings = ratings.tocsr()
            return ratings

    def als(self, database, **kwargs):
        from implicit.als import AlternatingLeastSquares
        opts = self.get_option('implicit', 'als', **kwargs)
        model = AlternatingLeastSquares(
            **opts
        )
        ratings = self.get_database(database)

        start_t = time.time()
        model.fit(ratings)
        elapsed = time.time() - start_t
        return elapsed

    def bpr(self, database, **kwargs):
        from implicit.bpr import BayesianPersonalizedRanking
        opts = self.get_option('implicit', 'bpr', **kwargs)
        model = BayesianPersonalizedRanking(
            **opts
        )
        ratings = self.get_database(database)

        start_t = time.time()
        model.fit(ratings)
        elapsed = time.time() - start_t
        return elapsed


class BuffaloLib(Benchmark):
    def __init__(self):
        super().__init__()

    def get_database(self, name):
        from buffalo.data.mm import MatrixMarketOptions
        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.validation = None
        data_opt.data.use_cache = True
        if name == 'ml20m':
            data_opt.input.main = '../tests/ml-20m/main'
        return data_opt

    def als(self, database, **kwargs):
        from buffalo.algo.als import ALS
        opts = self.get_option('buffalo', 'als', **kwargs)
        data_opt = self.get_database(database)
        als = ALS(opts, data_opt=data_opt)
        als.initialize()
        start_t = time.time()
        als.train()
        elapsed = time.time() - start_t
        return elapsed

    def bpr(self, database, **kwargs):
        from buffalo.algo.bpr import BPRMF
        opts = self.get_option('buffalo', 'bpr', **kwargs)
        data_opt = self.get_database(database)
        bpr = BPRMF(opts, data_opt=data_opt)
        bpr.initialize()
        start_t = time.time()
        bpr.train()
        elapsed = time.time() - start_t
        return elapsed
