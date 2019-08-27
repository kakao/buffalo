# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import fire
from models import ImplicitLib, BuffaloLib, LightfmLib, QmfLib, PysparkLib

from base import _get_elapsed_time, _print_table


def _performance(algo_name, database, lib):
    results = {}
    repeat = 3
    options = {'als': {'num_workers': 8,
                       'compute_loss_on_training': False,
                       'd': 40},
               'bpr': {'num_workers': 8,
                       'compute_loss_on_training': False,
                       'd': 40}
              }
    opt = options[algo_name]

    for d in [10, 20, 40, 80, 160]:
        opt['d'] = d
        elapsed, memory_usage = _get_elapsed_time(algo_name, database, lib, repeat, **opt)
        results[f'D={d}'] = elapsed
        results[f'DM={d}'] = memory_usage['max']
        results[f'DA={d}'] = memory_usage['avg']
        results[f'DB={d}'] = memory_usage['min']
        print(f'M={d} {elapsed} {memory_usage}')
        print(f'D={d} {elapsed}')

    opt['d'] = 20
    for num_workers in [1, 2, 4, 8, 16]:
        opt['num_workers'] = num_workers
        elapsed, memory_usage = _get_elapsed_time(algo_name, database, lib, repeat, **opt)
        results[f'T={num_workers}'] = elapsed
        results[f'TM={num_workers}'] = memory_usage['max']
        results[f'TA={num_workers}'] = memory_usage['avg']
        results[f'TB={num_workers}'] = memory_usage['min']
        print(f'M={num_workers} {elapsed} {memory_usage}')
        print(f'T={num_workers} {elapsed}')
    return results


def _memory(algo_name, database, lib):
    results = {}
    repeat = 3
    options = {'als': {'num_workers': 12,
                       'compute_loss_on_training': False,
                       'd': 32,
                       'num_iters': 2},
               'bpr': {'num_workers': 12,
                       'compute_loss_on_training': False,
                       'd': 32,
                       'num_iters': 2},
              }
    opt = options[algo_name]

    for batch_mb in [128, 256, 512, 1024, 2048, 4096]:
        if isinstance(lib, ImplicitLib):  # batch_mb effect to Buffalo only.
            break
        opt['batch_mb'] = batch_mb
        elapsed, memory_usage = _get_elapsed_time(algo_name, database, lib, repeat, **opt)
        results[f'E={batch_mb}'] = elapsed
        results[f'M={batch_mb}'] = memory_usage['max']
        results[f'A={batch_mb}'] = memory_usage['avg']
        results[f'B={batch_mb}'] = memory_usage['min']
        print(f'M={batch_mb} {elapsed} {memory_usage}')
    return results


def performance(algo_name, database, libs=['buffalo', 'implicit', 'lightfm', 'qmf', 'pyspark']):
    assert database in ['ml100k', 'ml20m', 'kakao_reco_730m', 'kakao_brunch_12m']
    assert algo_name in ['als', 'bpr']
    if isinstance(libs, str):
        libs = [libs]
    if algo_name == 'als':
        libs = [l for l in libs if l not in ['lightfm']]
    elif algo_name == 'bpr':
        libs = [l for l in libs if l not in ['pyspark']]
    R = {'buffalo': BuffaloLib,
         'implicit': ImplicitLib,
         'lightfm': LightfmLib,
         'qmf': QmfLib,
         'pyspark': PysparkLib}
    results = {l: _performance(algo_name, database, R[l]()) for l in libs}
    _print_table(results)


def memory(algo_name, database, libs=['buffalo', 'implicit']):
    assert database in ['ml100k', 'ml20m', 'kakao_reco_730m', 'kakao_brunch_12m']
    assert algo_name in ['als', 'bpr']
    if isinstance(libs, str):
        libs = [libs]
    R = {'buffalo': BuffaloLib,
         'implicit': ImplicitLib}
    results = {l: _memory(algo_name, database, R[l]()) for l in libs}
    _print_table(results)


if __name__ == '__main__':
    fire.Fire({'performance': performance,
               'memory': memory})
