# -*- coding: utf-8 -*-
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import datetime
import queue
import threading
import time

import h5py
import numpy as np
import psutil
from evaluate import evaluate_ranking_metrics

DB = {'kakao_reco_730m': './tmp/kakao_reco_730m.h5py',
      'ml20m': './tmp/ml20m.h5py',
      'ml100k': './tmp/ml100k.h5py',
      'kakao_brunch_12m': './tmp/kakao_brunch_12m.h5py'}


def collect_memory_usage(stop_event, result_queue):
    p = psutil.Process(os.getpid())
    data = []
    while not stop_event.is_set():
        data.append(p.memory_info().rss)
        time.sleep(5)
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
    data = context.parallelize(np.array([coo.row, coo.col, coo.data]).T)
    coo = None
    return spark.createDataFrame(data.map(lambda p: Row(row=int(p[0]),
                                                        col=int(p[1]),
                                                        data=float(p[2]))))


def get_buffalo_db(db):
    from buffalo.data.mm import MatrixMarket
    db_opt = BuffaloLib().get_database(db)
    db = MatrixMarket(db_opt)
    db.create()
    return db


class Benchmark(object):
    def get_option(self, lib_name, algo_name, **kwargs):
        if lib_name == 'buffalo':
            if algo_name == 'als':
                from buffalo.algo.options import ALSOption
                opt = ALSOption().get_default_option()
                opt.update({'d': kwargs.get('d', 100),
                            'optimizer': {True: 'manual_cg', False: 'ldlt'}.get(kwargs.get('use_cg', True)),
                            'num_iters': kwargs.get('num_iters', 10),
                            'num_cg_max_iters': 3,
                            'validation': kwargs.get('validation'),
                            'accelerator': kwargs.get('gpu', False),
                            'num_workers': kwargs.get('num_workers', 10),
                            'compute_loss_on_training': kwargs.get('compute_loss_on_training', False)})
                return opt
            if algo_name == 'bpr':
                from buffalo.algo.options import BPRMFOption
                opt = BPRMFOption().get_default_option()
                opt.update({'d': kwargs.get('d', 100),
                            'lr': kwargs.get('lr', 0.05),
                            'validation': kwargs.get('validation'),
                            'num_iters': kwargs.get('num_iters', 10),
                            'num_workers': kwargs.get('num_workers', 10),
                            'compute_loss_on_training': kwargs.get('compute_loss_on_training', False)})
                return opt
            if algo_name == 'warp':
                from buffalo.algo.options import WARPOption
                opt = WARPOption().get_default_option()
                opt.update({'d': kwargs.get('d', 100),
                            'lr': kwargs.get('lr', 0.05),
                            'validation': kwargs.get('validation'),
                            'num_iters': kwargs.get('num_iters', 10),
                            'max_trials': 100,
                            'num_workers': kwargs.get('num_workers', 10),
                            'compute_loss_on_training': kwargs.get('compute_loss_on_training', False)})
                return opt
        elif lib_name == 'implicit':
            if algo_name == 'als':
                return {'factors': kwargs.get('d', 100),
                        'dtype': np.float32,
                        'use_native': True,
                        'use_gpu': kwargs.get('gpu', False),
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
            if algo_name == 'warp':
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
                        'intermediateStorageLevel': 'MEMORY_ONLY',
                        'finalStorageLevel': 'MEMORY_ONLY',
                        'ratingCol': 'data'}

    def run(self, func, *args, **kwargs):
        stop_event = threading.Event()
        result_queue = queue.Queue()
        t = threading.Thread(target=collect_memory_usage, args=(stop_event, result_queue))
        t.start()
        time.sleep(0.1)  # context switching
        start_t = time.time()
        if kwargs.get('iterable'):
            iterable = kwargs.get('iterable')
            kwargs.pop('iterable')
            for data in iterable:
                func(data, **kwargs)
        else:
            func(*args, **kwargs)
        elapsed = time.time() - start_t
        stop_event.set()
        t.join()
        memory_usage = result_queue.get(block=True, timeout=10)
        return elapsed, {'min': min(memory_usage) / 1024 / 1024.0,
                         'avg': sum(memory_usage) / len(memory_usage) / 1024 / 1024.0,
                         'max': max(memory_usage) / 1024 / 1024.0}


class ImplicitLib(Benchmark):
    def __init__(self):
        super().__init__()

    def get_database(self, name, **kwargs):
        if name in ['ml20m', 'ml100k', 'kakao_reco_730m', 'kakao_brunch_12m']:
            db = h5py.File(DB[name], 'r')
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
        if kwargs.get('return_instance_before_train'):
            return (model, ratings)

        elapsed, mem_info = self.run(model.fit, ratings)
        if kwargs.get('return_instance'):
            return model
        model = None
        return elapsed, mem_info

    def bpr(self, database, **kwargs):
        from implicit.bpr import BayesianPersonalizedRanking
        opts = self.get_option('implicit', 'bpr', **kwargs)
        model = BayesianPersonalizedRanking(
            **opts
        )
        ratings = self.get_database(database, **kwargs)
        if kwargs.get('return_instance_before_train'):
            return (model, ratings)

        elapsed, mem_info = self.run(model.fit, ratings)
        if kwargs.get('return_instance'):
            return model
        model = None
        return elapsed, mem_info

    def most_similar(self, keys, **kwargs):
        model = kwargs['model']
        kwargs.pop('model')
        elapsed, mem_info = self.run(model.similar_items, keys, **kwargs)
        return elapsed, mem_info


class BuffaloLib(Benchmark):
    def __init__(self):
        super().__init__()

    def get_database(self, name, **kwargs):
        from buffalo.data.mm import MatrixMarketOptions
        data_opt = MatrixMarketOptions().get_default_option()
        if kwargs.get('validation', None) is None:
            data_opt.validation = None
        data_opt.data.use_cache = True
        data_opt.data.batch_mb = kwargs.get('batch_mb', 1024)
        if name == 'ml20m':
            data_opt.data.path = DB[name]
            data_opt.input.main = '../tests/ext/ml-20m/main'
        elif name == 'ml100k':
            data_opt.data.path = DB[name]
            data_opt.input.main = '../tests/ext/ml-100k/main'
        elif name == 'kakao_reco_730m':
            data_opt.data.path = DB[name]
            data_opt.data.tmp_dir = './tmp/'
            data_opt.input.main = '../tests/ext/kakao-reco-730m/main'
        elif name == 'kakao_brunch_12m':
            data_opt.data.path = DB[name]
            data_opt.data.tmp_dir = './tmp/'
            data_opt.input.main = '../tests/ext/kakao-brunch-12m/main'
        return data_opt

    def als(self, database, **kwargs):
        from buffalo.algo.als import ALS
        opts = self.get_option('buffalo', 'als', **kwargs)
        data_opt = self.get_database(database, **kwargs)
        als = ALS(opts, data_opt=data_opt)
        als.initialize()
        if kwargs.get('return_instance_before_train'):
            return als
        elapsed, mem_info = self.run(als.train)
        if kwargs.get('return_instance'):
            return als
        als = None
        return elapsed, mem_info

    def validation(self, algo_name, database, **kwargs):
        inst = getattr(self, algo_name)(database, return_instance=True, **kwargs)
        ret = inst.get_validation_results()  # same as below
        for p in ['error', 'rmse']:
            ret.pop(p)
        return ret
        inst.data._prepare_validation_data()
        K = kwargs.get('validation', {}).get('topk', 10)
        userids = list(set(inst.data.handle['vali']['row'][::]))
        recs = []
        for user in userids:
            seen = inst.data.vali_data['validation_seen'].get(user, set())
            user_str = inst.data.handle['idmap']['rows'][user].decode('utf-8')
            topn = inst.topk_recommendation(user_str, topk=K + len(seen))
            topn = [inst._idmanager.itemid_map[t] for t in topn]
            topn = [t for t in topn if t not in seen][:K]
            recs.append((user, topn))
        return evaluate_ranking_metrics(recs, K, inst.data.vali_data, inst.data.header['num_items'])

    def bpr(self, database, **kwargs):
        from buffalo.algo.bpr import BPRMF
        opts = self.get_option('buffalo', 'bpr', **kwargs)
        data_opt = self.get_database(database, **kwargs)
        bpr = BPRMF(opts, data_opt=data_opt)
        bpr.initialize()
        if kwargs.get('return_instance_before_train'):
            return bpr
        elapsed, mem_info = self.run(bpr.train)
        if kwargs.get('return_instance'):
            return bpr
        bpr = None
        return elapsed, mem_info

    def warp(self, database, **kwargs):
        from buffalo.algo.warp import WARP
        opts = self.get_option('buffalo', 'warp', **kwargs)
        data_opt = self.get_database(database, **kwargs)
        warp = WARP(opts, data_opt=data_opt)
        warp.initialize()
        if kwargs.get('return_instance_before_train'):
            return warp
        elapsed, mem_info = self.run(warp.train)
        if kwargs.get('return_instance'):
            return warp
        warp = None
        return elapsed, mem_info

    def most_similar(self, keys, **kwargs):
        model = kwargs['model']
        kwargs.pop('model')
        elapsed, mem_info = self.run(model.most_similar, keys, **kwargs)
        return elapsed, mem_info


class LightfmLib(Benchmark):
    def __init__(self):
        super().__init__()

    def get_database(self, name, **kwargs):
        if name in ['ml20m', 'ml100k', 'kakao_reco_730m', 'kakao_brunch_12m']:
            db = h5py.File(DB[name], 'r')
            ratings = db_to_coo(db)
            db.close()
            return ratings

    def als(self, database, **kwargs):
        raise NotImplementedError

    def bpr(self, database, **kwargs):
        from lightfm import LightFM
        opts = self.get_option('lightfm', 'bpr', **kwargs)
        data = self.get_database(database, **kwargs)
        bpr = LightFM(loss='bpr',
                      no_components=kwargs.get('d'))
        elapsed, mem_info = self.run(bpr.fit, data, **opts)
        if kwargs.get('return_instance'):
            return bpr
        bpr = None
        return elapsed, mem_info

    def validation(self, algo_name, database, **kwargs):
        K = kwargs.get('validation', {}).get('topk', 10)
        inst = getattr(self, algo_name)(database, return_instance=True, **kwargs)
        db = get_buffalo_db(database)
        db._prepare_validation_data()
        userids = list(set(db.handle['vali']['row'][::]))
        itemids = list(range(db.handle['idmap']['cols'].shape[0]))
        recs = []
        for user in userids:
            topn = np.argsort(-inst.predict(user, itemids))
            topn = [t for t in topn if t not in db.vali_data['validation_seen'].get(user, set())][:K]
            recs.append((user, topn))
        return evaluate_ranking_metrics(recs, K, db.vali_data, db.header['num_items'])

    def warp(self, database, **kwargs):
        from lightfm import LightFM
        opts = self.get_option('lightfm', 'warp', **kwargs)
        data = self.get_database(database, **kwargs)
        warp = LightFM(loss='warp',
                       learning_schedule='adagrad',
                       no_components=kwargs.get('d'),
                       max_sampled=100)
        elapsed, mem_info = self.run(warp.fit, data, **opts)
        if kwargs.get('return_instance'):
            return warp
        warp = None
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
        if name in ['kakao_brunch_12m']:
            db_path = f'./qmf.{name}.dataset'
            num_header_lines = 4
            if not os.path.isfile(db_path):
                with open('../tests/ext/kakao-brunch-12m/main') as fin:
                    for i in range(num_header_lines):
                        _ = fin.readline()
                    with open(db_path, 'w') as fout:
                        for line in fin:
                            fout.write(line)
            return os.path.abspath(db_path)
        elif name == 'kakao_reco_730m':
            db_path = f'./qmf.{name}.dataset'
            num_header_lines = 4
            if not os.path.isfile(db_path):
                with open('../tests/ext/kakao-reco-730m/main') as fin:
                    for i in range(num_header_lines):
                        _ = fin.readline()
                    with open(db_path, 'w') as fout:
                        for line in fin:
                            fout.write(line)
            return os.path.abspath(db_path)

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
        if name in ['ml20m', 'ml100k', 'kakao_reco_730m', 'kakao_brunch_12m']:
            db = h5py.File(DB[name], 'r')
            ratings = db_to_dataframe(db, kwargs.get('spark'), kwargs.get('context'))
            db.close()
            return ratings

    def als(self, database, **kwargs):
        from pyspark import SparkConf, SparkContext
        from pyspark.ml.recommendation import ALS
        from pyspark.sql import SparkSession
        opts = self.get_option('pyspark', 'als', **kwargs)
        conf = SparkConf()\
            .setAppName("pyspark")\
            .setMaster('local[%s]' % kwargs.get('num_workers'))\
            .set('spark.local.dir', './tmp/')\
            .set('spark.worker.cleanup.enabled', 'true')\
            .set('spark.driver.memory', '32G')
        context = SparkContext(conf=conf)
        context.setLogLevel('WARN')
        spark = SparkSession(context)
        data = self.get_database(database, spark=spark, context=context)
        print(opts)
        als = ALS(**opts)
        elapsed, memory_usage = self.run(als.fit, data)
        spark.stop()
        return elapsed, memory_usage

    def bpr(self, database, **kwargs):
        raise NotImplementedError
