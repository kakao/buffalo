# -*- coding: utf-8 -*-
import fire
from tabulate import tabulate

from models import ImplicitLib, BuffaloLib


def _get_elapsed_time(algo_name, database, lib, repeat, **options):
    elapsed = []
    for i in range(repeat):
        e = getattr(lib, algo_name)(database, **options)
        elapsed.append(e)
    elapsed = sum(elapsed) / len(elapsed)
    return elapsed


def _test1(algo_name, database, lib):
    results = {}
    repeat = 3
    options = {'als': {'num_workers': 8,
                       'compute_loss_on_training': False,
                       'use_conjugate_gradient': True,
                       'd': 40},
               'bpr': {'num_workers': 8,
                       'compute_loss_on_training': False,
                       'd': 40}
              }
    opt = options[algo_name]

    for d in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]:
        opt['d'] = d
        elapsed = _get_elapsed_time(algo_name, database, lib, repeat, **opt)
        results[f'D={d}'] = elapsed
        print(f'D={d} {elapsed}')

    opt['d'] = 32
    for num_workers in [1, 2, 4, 6, 12]:
        opt['num_workers'] = num_workers
        elapsed = _get_elapsed_time(algo_name, database, lib, repeat, **opt)
        results[f'T={num_workers}'] = elapsed
        print(f'T={num_workers} {elapsed}')
    return results


def benchmark1(algo_name, database, libs=['buffalo', 'implicit']):
    assert algo_name in ['als', 'bpr']
    if isinstance(libs, str):
        libs = [libs]
    R = {'buffalo': BuffaloLib,
         'implicit': ImplicitLib}
    results = {l: _test1(algo_name, database, R[l]()) for l in libs}

    for f in ['D', 'T']:
        table = []
        for lib_name in libs:
            data = sorted([(k, v) for k, v in results[lib_name].items()
                        if k.startswith(f)],
                        key=lambda x: (len(x[0]), x[0]))
            rows = [v for _, v in data]
            headers = ['method'] + [k for k, _ in data]
            table.append([lib_name] + rows)
        print(tabulate(table, headers=headers, tablefmt="github"))
        print('')


def _test2(algo_name, database, lib):
    results = {}
    repeat = 1
    options = {'als': {'num_workers': 12,
                       'compute_loss_on_training': False,
                       'use_conjugate_gradient': True,
                       'd': 32},
               'bpr': {'num_workers': 12,
                       'compute_loss_on_training': False,
                       'd': 32}
              }
    opt = options[algo_name]

    for batch_mb in [1024, 2048, 4092]:
        opt['batch_mb'] = batch_mb
        results[f'M={batch_mb}'] = batch_mb
        elapsed = _get_elapsed_time(algo_name, database, lib, repeat, **opt)
        print(f'M={batch_mb} {elapsed}')
    return results


def benchmark2(algo_name, database, libs=['buffalo', 'implicit']):
    assert algo_name in ['als', 'bpr']
    if isinstance(libs, str):
        libs = [libs]
    R = {'buffalo': BuffaloLib,
         'implicit': ImplicitLib}
    results = {l: _test2(algo_name, database, R[l]()) for l in libs}

    for f in ['M']:
        table = []
        for lib_name in libs:
            data = sorted([(k, v) for k, v in results[lib_name].items()
                        if k.startswith(f)],
                        key=lambda x: (len(x[0]), x[0]))
            rows = [v for _, v in data]
            headers = ['method'] + [k for k, _ in data]
            table.append([lib_name] + rows)
        print(tabulate(table, headers=headers, tablefmt="github"))
        print('')


if __name__ == '__main__':
    fire.Fire({'benchmark1': benchmark1,
               'benchmark2': benchmark2})
