# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import time
import queue
import threading

import h5py
import psutil
import numpy as np
from implicit.datasets.movielens import get_movielens


def collect_memory_usage(stop_event, result_queue):
    p = psutil.Process(os.getpid())
    data = []
    while not stop_event.is_set():
        time.sleep(5)
        data.append(p.memory_info().rss)
    result_queue.put(data)


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

    def run(self, func, *args):
        stop_event = threading.Event()
        result_queue = queue.Queue()
        t = threading.Thread(target=collect_memory_usage, args=(stop_event, result_queue))
        t.start()
        start_t = time.time()
        func(*args)
        elapsed = time.time() - start_t
        stop_event.set()
        t.join()
        memory_usage = result_queue.get()
        model = None
        return elapsed, {'min': min(memory_usage) / 1024 / 1024.0,
                         'avg': sum(memory_usage) / len(memory_usage) / 1024 / 1024.0,
                         'max': max(memory_usage) / 1024 / 1024.0}


class ImplicitLib(Benchmark):
    def __init__(self):
        super().__init__()

    def get_database(self, name, **kwargs):
        if name == 'ml20m':
            _, ratings = get_movielens('20m')
            ratings.data = np.ones(len(ratings.data))
            ratings = ratings.tocsr()
            return ratings
        elif name == 'kakao_reco_medium':
            db = h5py.File('kakao_reco_medium.h5py')
            ratings = db_to_coo(db)
            return ratings

    def als(self, database, **kwargs):
        from implicit.als import AlternatingLeastSquares
        opts = self.get_option('implicit', 'als', **kwargs)
        model = AlternatingLeastSquares(
            **opts
        )
        ratings = self.get_database(database, **kwargs)

        elasepd, mem_info = self.run(model.fit, ratings)
        model = None
        return elapsed, mem_info

    def bpr(self, database, **kwargs):
        from implicit.bpr import BayesianPersonalizedRanking
        opts = self.get_option('implicit', 'bpr', **kwargs)
        model = BayesianPersonalizedRanking(
            **opts
        )
        ratings = self.get_database(database, **kwargs)

        elapsed, mem_info = self.run(model.fit, ratings)
        model = None
        return elapsed, mem_info


class BuffaloLib(Benchmark):
    def __init__(self):
        super().__init__()

    def get_database(self, name, **kwargs):
        from buffalo.data.mm import MatrixMarketOptions
        data_opt = MatrixMarketOptions().get_default_option()
        data_opt.validation = None
        data_opt.data.use_cache = True
        data_opt.data.batch_mb = kwargs.get('batch_mb', 1024)
        if name == 'ml20m':
            data_opt.input.main = '../tests/ml-20m/main'
        elif name == 'kakao_reco_medium':
            data_opt.data.path = 'kakao_reco_medium.h5py'
            data_opt.data.tmp_dir = './tmp/'
            data_opt.input.main = '../tests/kakao_reco_medium/main'
        return data_opt

    def als(self, database, **kwargs):
        from buffalo.algo.als import ALS
        opts = self.get_option('buffalo', 'als', **kwargs)
        data_opt = self.get_database(database, **kwargs)
        als = ALS(opts, data_opt=data_opt)
        als.initialize()
        elapsed, mem_info = self.run(als.train)
        als = None
        return elapsed, mem_info

    def bpr(self, database, **kwargs):
        from buffalo.algo.bpr import BPRMF
        opts = self.get_option('buffalo', 'bpr', **kwargs)
        data_opt = self.get_database(database, **kwargs)
        bpr = BPRMF(opts, data_opt=data_opt)
        bpr.initialize()
        elapsed, mem_info = self.run(bpr.train)
        bpr = None
        return elapsed, mem_info


def db_to_coo(db):
    from scipy.sparse import coo_matrix
    V = db.attrs['num_items']
    U = db.attrs['num_users']
    col = db['rowwise']['key'][::]
    data = db['rowwise']['val'][::]
    indptr = db['rowwise']['indptr'][::]
    real_size = indptr[-1]
    col = col[:real_size]
    data = data[:real_size]
    row = np.zeros(col.shape, dtype=np.int32)
    start, row_id = 0, 0
    for end in indptr:
        row[start:end] = row_id
        row_id += 1
        start = end
    return coo_matrix((data, (row, col)), (U, V))
