# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import time
import queue
import datetime
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
    coo = coo_matrix((data, (row, col)), (U, V))
    data, row, col = None, None, None
    return coo


def db_to_dataframe(db, spark, context):
    from pyspark.sql import Row
    coo = db_to_coo(db)
    data = context.parallelize(np.array([coo.row, coo.col, coo.data]).T,
                               numSlices=len(coo.row) / 1024)
    coo = None
    return spark.createDataFrame(data.map(lambda p: Row(row=int(p[0]),
                                                        col=int(p[1]),
                                                        data=float(p[2]))))


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
        elif lib_name == 'lightfm':
            if algo_name == 'bpr':
                return {'epochs': kwargs.get('num_iters', 10),
                        'verbose': True,
                        'num_threads': kwargs.get('num_workers', 10)}
        elif lib_name == 'pyspark':
            if algo_name == 'als':
                return {'maxIter': kwargs.get('num_iters', 10),
                        'rank': kwargs.get('d', 100),
                        'alpha': 8,
                        'implicitPrefs': True,
                        'userCol': 'row',
                        'itemCol': 'col',
                        # 'intermediateStorageLevel': 'MEMORY_ONLY',
                        # 'finalStorageLevel': 'MEMORY_ONLY',
                        'ratingCol': 'data'}

    def run(self, func, *args, **kwargs):
        stop_event = threading.Event()
        result_queue = queue.Queue()
        t = threading.Thread(target=collect_memory_usage, args=(stop_event, result_queue))
        t.start()
        start_t = time.time()
        func(*args, **kwargs)
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
        if name in ['ml20m', 'ml100k']:
            _, ratings = get_movielens({'ml20m': '20m', 'ml100k': '100k'}.get(name))
            ratings.data = np.ones(len(ratings.data))
            ratings = ratings.tocsr()
            return ratings
        elif name == 'kakao_reco_medium':
            db = h5py.File('kakao_reco_medium.h5py')
            ratings = db_to_coo(db)
            db.close()
            return ratings

    def als(self, database, **kwargs):
        from implicit.als import AlternatingLeastSquares
        opts = self.get_option('implicit', 'als', **kwargs)
        model = AlternatingLeastSquares(
            **opts
        )
        ratings = self.get_database(database, **kwargs)

        elapsed, mem_info = self.run(model.fit, ratings)
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
            data_opt.data.path = 'ml20m.h5py'
            data_opt.input.main = '../tests/ml-20m/main'
        elif name =='ml100k':
            data_opt.data.path = 'ml100k.h5py'
            data_opt.input.main = '../tests/ml-100k/main'
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


class LightfmLib(Benchmark):
    def __init__(self):
        super().__init__()

    def get_database(self, name, **kwargs):
        if name == 'ml20m':
            db = h5py.File('ml20m.h5py')
            ratings = db_to_coo(db)
            db.close()
            return ratings
        if name == 'ml100k':
            db = h5py.File('ml100k.h5py')
            ratings = db_to_coo(db)
            db.close()
            return ratings
        elif name == 'kakao_reco_medium':
            db = h5py.File('kakao_reco_medium.h5py')
            ratings = db_to_coo(db)
            db.close()
            return ratings

    def als(self, database, **kwargs):
        raise NotImplemented

    def bpr(self, database, **kwargs):
        from lightfm import LightFM
        opts = self.get_option('lightfm', 'bpr', **kwargs)
        data = self.get_database(database, **kwargs)
        bpr = LightFM(no_components=kwargs.get('num_workers'),
                      max_sampled=1)
        elapsed, mem_info = self.run(bpr.fit, data, data, **opts)
        bpr = None
        return elapsed, mem_info


class QmfLib(Benchmark):
    def __init__(self):
        super().__init__()
        self.bin_root = '../../qmf.git/obj/bin'

    def get_database(self, name, **kwargs):
        if name in ['ml20m', 'ml100k']:
            db_path = f'./qmf.{name}.dataset'
            num_header_lines = 4
            if not os.path.isfile(db_path):
                with open('../tests/%s/main' % {'ml20m': 'ml-20m', 'ml100k': 'ml-100k'}.get(name)) as fin:
                    for i in range(num_header_lines):
                        _ = fin.readline()
                    with open(db_path, 'w') as fout:
                        for line in fin:
                            fout.write(line)
            return os.path.abspath(db_path)
        elif name == 'kakao_reco_medium':
            return NotImplemented

    def als(self, database, **kwargs):
        import subprocess
        train_ds = self.get_database(database, **kwargs)
        d = kwargs.get('d')
        num_iters = kwargs.get('num_iters', 10)
        num_workers = kwargs.get('num_workers')
        cmd = ['./wals',
               f'--train_dataset={train_ds}',
               '--user_factors=/dev/null',
               '--item_factors=/dev/null',
               f'--nfactors={d}',
               f'--nepochs={num_iters}',
               f'--nthreads={num_workers}']
        stop_event = threading.Event()
        result_queue = queue.Queue()
        t = threading.Thread(target=collect_memory_usage, args=(stop_event, result_queue))
        t.start()
        ret = subprocess.run(cmd, cwd=self.bin_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stop_event.set()
        t.join()
        memory_usage = result_queue.get()

        start_t, end_t = None, None
        for line in ret.stderr.decode('utf-8').split('\n'):
            if line.endswith('training'):
                hms = line.split()[1]
                start_t = datetime.datetime.strptime(hms, '%H:%M:%S.%f')
            elif line.endswith('saving model output'):
                hms = line.split()[1]
                end_t = datetime.datetime.strptime(hms, '%H:%M:%S.%f')
        elapsed = (end_t - start_t).seconds + (end_t - start_t).microseconds / 1000 / 1000
        return elapsed, {'min': min(memory_usage) / 1024 / 1024.0,
                         'avg': sum(memory_usage) / len(memory_usage) / 1024 / 1024.0,
                         'max': max(memory_usage) / 1024 / 1024.0}

    def bpr(self, database, **kwargs):
        import subprocess
        train_ds = self.get_database(database, **kwargs)
        d = kwargs.get('d')
        num_iters = kwargs.get('num_iters', 10)
        num_workers = kwargs.get('num_workers')
        cmd = ['./bpr',
               f'--train_dataset={train_ds}',
               '--user_factors=/dev/null',
               '--item_factors=/dev/null',
               '--num_negative_samples=1',
               '--eval_num_neg=0',
               f'--nfactors={d}',
               f'--nepochs={num_iters}',
               f'--nthreads={num_workers}']
        stop_event = threading.Event()
        result_queue = queue.Queue()
        t = threading.Thread(target=collect_memory_usage, args=(stop_event, result_queue))
        t.start()
        ret = subprocess.run(cmd, cwd=self.bin_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stop_event.set()
        t.join()
        memory_usage = result_queue.get()

        start_t, end_t = None, None
        for line in ret.stderr.decode('utf-8').split('\n'):
            if line.endswith('training'):
                hms = line.split()[1]
                start_t = datetime.datetime.strptime(hms, '%H:%M:%S.%f')
            elif line.endswith('saving model output'):
                hms = line.split()[1]
                end_t = datetime.datetime.strptime(hms, '%H:%M:%S.%f')
        elapsed = (end_t - start_t).seconds + (end_t - start_t).microseconds / 1000 / 1000
        return elapsed, {'min': min(memory_usage) / 1024 / 1024.0,
                         'avg': sum(memory_usage) / len(memory_usage) / 1024 / 1024.0,
                         'max': max(memory_usage) / 1024 / 1024.0}


class PysparkLib(Benchmark):
    def __init__(self):
        super().__init__()

    def get_database(self, name, **kwargs):
        if name in ['ml20m', 'ml100k']:
            db = h5py.File({'ml20m': 'ml20m.h5py',
                            'ml100k': 'ml100k.h5py'}.get(name))
            ratings = db_to_dataframe(db, kwargs.get('spark'), kwargs.get('context'))
            db.close()
            return ratings
        elif name == 'kakao_reco_medium':
            db = h5py.File('kakao_reco_medium.h5py')
            ratings = db_to_dataframe(db, kwargs.get('spark'), kwargs.get('context'))
            db.close()
            return ratings

    def als(self, database, **kwargs):
        from pyspark.sql import SparkSession
        from pyspark.ml.recommendation import ALS
        from pyspark import SparkConf, SparkContext
        opts = self.get_option('pyspark', 'als', **kwargs)
        conf = SparkConf()\
               .setAppName("pyspark")\
               .setMaster('local[%s]' % kwargs.get('num_workers'))\
               .set('spark.cores.max', 1)\
               .set('spark.local.dir', './tmp/')\
               .set('spark.driver.memory', '32G')
        context = SparkContext(conf=conf)
        context.setLogLevel('INFO')
        spark = SparkSession(context)
        data = self.get_database(database, spark=spark, context=context)
        print(opts)
        als = ALS(**opts)
        elapsed, memory_usage = self.run(als.fit, data)
        spark.stop()
        return elapsed, memory_usage

    def bpr(self, database, **kwargs):
        raise NotImplemented
