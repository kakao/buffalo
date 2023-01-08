# -*- coding: utf-8 -*-
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import fire
import numpy as np
from base import _get_elapsed_time, _print_table
from models import BuffaloLib, ImplicitLib
from n2 import HnswIndex

from buffalo.parallel import ParALS, ParBPRMF


def _buffalo(algo_name, database):
    repeat = 3
    options = {'als': {'num_workers': 4,
                       'compute_loss_on_training': False,
                       'd': 32,
                       'num_iters': 10},
               'bpr': {'num_workers': 4,
                       'compute_loss_on_training': False,
                       'd': 32,
                       'num_iters': 100}}
    opt = options[algo_name]

    # linear
    if algo_name == 'als':
        PAR = ParALS
        model = BuffaloLib().als(database, return_instance_before_train=True, **opt)
    elif algo_name == 'bpr':
        PAR = ParBPRMF
        model = BuffaloLib().bpr(database, return_instance_before_train=True, **opt)
    model.train()
    model.build_itemid_map()
    model.normalize('item')

    # parallel
    par = PAR(model)

    # ann
    index = HnswIndex(model.P.shape[1])
    for f in model.P:
        index.add_data(f)
    index.build(n_threads=4)
    index.save('bm_n2.bin')

    ann = PAR(model)
    ann.set_hnsw_index('bm_n2.bin', 'item')

    total_queries = 10000
    keys = model._idmanager.itemids[::][:total_queries]
    print('Total queries: %s' % len(keys))
    results = {}
    nn_opts = {'topk': 10}
    for p, m in [('S', model), ('P', par), ('A', ann)]:
        results[p] = {}
        opt = nn_opts.copy()
        if not isinstance(m, PAR):
            opt['iterable'] = keys
        for num_workers in [1, 2, 4]:
            if isinstance(m, PAR):
                m.num_workers = num_workers
            else:
                m.opt.num_workers = num_workers
            opt['model'] = m
            elapsed, memory_usage = _get_elapsed_time('most_similar',
                                                      keys,
                                                      BuffaloLib(), repeat, **opt)
            s = elapsed / len(keys)
            results[p][f'S={num_workers}'] = s
            results[p][f'E={num_workers}'] = elapsed
            results[p][f'M={num_workers}'] = memory_usage['max']
            results[p][f'A={num_workers}'] = memory_usage['avg']
            results[p][f'B={num_workers}'] = memory_usage['min']
            print(f'{p}M={num_workers} {elapsed} {memory_usage}')
    return results


def buffalo(algo_name='als', database='kakao_brunch_12m'):
    assert database in ['ml100k', 'ml20m', 'kakao_reco_730m', 'kakao_brunch_12m']
    assert algo_name in ['als', 'bpr']
    results = _buffalo(algo_name, database)
    _print_table(results)


def _implicit(algo_name, database):
    repeat = 3
    options = {'als': {'num_workers': 4,
                       'compute_loss_on_training': False,
                       'd': 32,
                       'num_iters': 10},
               'bpr': {'num_workers': 4,
                       'compute_loss_on_training': False,
                       'd': 32,
                       'num_iters': 100}}
    opt = options[algo_name]

    # linear
    if algo_name == 'als':
        model, ratings = ImplicitLib().als(database, return_instance_before_train=True, **opt)
    elif algo_name == 'bpr':
        model, ratings = ImplicitLib().bpr(database, return_instance_before_train=True, **opt)
    model.fit(ratings)

    total_queries = 10000
    keys = sorted(np.arange(total_queries))
    print('Total queries: %s' % len(keys))
    results = {}
    nn_opts = {'N': 10}
    for p, m in [('S', model)]:
        results[p] = {}
        opt = nn_opts.copy()
        opt['iterable'] = keys
        for num_workers in [1]:
            opt['model'] = m
            elapsed, memory_usage = _get_elapsed_time('most_similar',
                                                      keys,
                                                      ImplicitLib(), repeat, **opt)
            s = elapsed / len(keys)
            results[p][f'S={num_workers}'] = s
            results[p][f'E={num_workers}'] = elapsed
            results[p][f'M={num_workers}'] = memory_usage['max']
            results[p][f'A={num_workers}'] = memory_usage['avg']
            results[p][f'B={num_workers}'] = memory_usage['min']
            print(f'{p}M={num_workers} {elapsed} {memory_usage}')
    return results


def implicit(algo_name='als', database='kakao_brunch_12m'):
    assert database in ['ml100k', 'ml20m', 'kakao_reco_730m', 'kakao_brunch_12m']
    assert algo_name in ['als', 'bpr']
    results = _implicit(algo_name, database)
    _print_table(results)


if __name__ == '__main__':
    fire.Fire({'buffalo': buffalo,
               'implicit': implicit})
