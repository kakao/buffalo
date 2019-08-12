# -*- coding: utf-8 -*-
import fire
from tabulate import tabulate

from models import ImplicitLib, BuffaloLib


def _get_elapsed_time(algo_name, database, lib, repeat, **options):
    elapsed = []
    mem_info = []
    for i in range(repeat):
        e, m = getattr(lib, algo_name)(database, **options)
        elapsed.append(e)
        mem_info.append(m)
    elapsed = sum(elapsed) / len(elapsed)
    mem_info = {'min': min([m['min'] for m in mem_info]),
                'max': max([m['max'] for m in mem_info]),
                'avg': sum([m['avg'] for m in mem_info]) / len(mem_info)}
    return elapsed, mem_info


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
        elapsed, _ = _get_elapsed_time(algo_name, database, lib, repeat, **opt)
        results[f'D={d}'] = elapsed
        print(f'D={d} {elapsed}')

    opt['d'] = 32
    for num_workers in [1, 2, 4, 6, 12]:
        opt['num_workers'] = num_workers
        elapsed, _ = _get_elapsed_time(algo_name, database, lib, repeat, **opt)
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
    repeat = 3
    options = {'als': {'num_workers': 12,
                       'compute_loss_on_training': False,
                       'use_conjugate_gradient': True,
                       'd': 32,
                       'num_iters': 2},
               'bpr': {'num_workers': 12,
                       'compute_loss_on_training': False,
                       'd': 32,
                       'num_iters': 2},
              }
    opt = options[algo_name]

    for batch_mb in [128, 256, 512, 1024]:
        opt['batch_mb'] = batch_mb
        elapsed, memory_usage = _get_elapsed_time(algo_name, database, lib, repeat, **opt)
        results[f'E={batch_mb}'] = elapsed
        results[f'M={batch_mb}'] = memory_usage['max']
        results[f'A={batch_mb}'] = memory_usage['avg']
        results[f'B={batch_mb}'] = memory_usage['min']
        print(f'M={batch_mb} {elapsed} {memory_usage}')
    return results


def benchmark2(algo_name, database, libs=['buffalo', 'implicit']):
    assert algo_name in ['als', 'bpr']
    if isinstance(libs, str):
        libs = [libs]
    R = {'buffalo': BuffaloLib,
         'implicit': ImplicitLib}
    results = {l: _test2(algo_name, database, R[l]()) for l in libs}

    for f in ['M', 'A', 'B']:
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
