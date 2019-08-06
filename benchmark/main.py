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


def _test(algo_name, database, lib):
    results = {}
    repeat = 1
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

    opt['d'] = 40
    for num_workers in [1, 2, 4, 8, 16]:
        opt['num_workers'] = num_workers
        elapsed = _get_elapsed_time(algo_name, database, lib, repeat, **opt)
        results[f'T={num_workers}'] = elapsed
        print(f'T={num_workers} {elapsed}')
    return results


def test(algo_name, database):
    assert algo_name in ['als', 'bpr']
    results = {'buffalo': _test(algo_name, database, BuffaloLib()),
               'implicit': _test(algo_name, database, ImplicitLib())}

    for f in ['D', 'T']:
        table = []
        for lib_name in ['buffalo', 'implicit']:
            data = sorted([(k, v) for k, v in results[lib_name].items()
                        if k.startswith(f)],
                        key=lambda x: x[0])
            rows = [v for _, v in data]
            headers = ['method'] + [k for k, _ in data]
            table.append([lib_name] + rows)
        print(tabulate(table, headers=headers, tablefmt="github"))
        print('')


if __name__ == '__main__':
    fire.Fire({'run': test})
