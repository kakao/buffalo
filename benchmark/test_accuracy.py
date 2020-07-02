# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import fire
from models import BuffaloLib, LightfmLib

from base import _print_table


def _buffalo_validation(algo_name, database):
    options = {
        'als': {'num_workers': 8,
                'compute_loss_on_training': False,
                'batch_mb': 4098,
                'd': 40},
        'bpr': {'num_workers': 8,
                'batch_mb': 4098,
                'compute_loss_on_training': False,
                'd': 40},
        'warp': {'num_workers': 8,
                 'batch_mb': 4098,
                 'compute_loss_on_training': False,
                 'd': 40}
    }
    opt = options[algo_name]
    opt.update({'return_instance': True, 'validation': {'topk': 10},
                'num_iters': 100})

    blib = BuffaloLib()
    inst = getattr(blib, algo_name)(database, **opt)
    ret = inst.get_validation_results()
    return ret


def compare_warp_bpr(database):
    assert database in ['kakao_reco_730m', 'ml20m', 'ml100k', 'kakao_brunch_12m']
    rets = {}
    rets['bprmf'] = _buffalo_validation('bpr', database)
    rets['warp'] = _buffalo_validation('warp', database)
    _print_table(rets)


def _get_validation_score(algo_name, lib, database):
    options = {
        'als': {'num_workers': 8,
                'compute_loss_on_training': False,
                'batch_mb': 4098,
                'evaluation_on_learning': False,
                'validation': {'topk': 10},
                'd': 40},
        'bpr': {'num_workers': 8,
                'batch_mb': 4098,
                'compute_loss_on_training': False,
                'num_iters': 100,
                'validation': {'topk': 10},
                'd': 40},
        'warp': {'num_workers': 8,
                 'batch_mb': 4098,
                 'compute_loss_on_training': False,
                 'num_iters': 100,
                 'validation': {'topk': 10},
                 'd': 40}
    }
    opt = options[algo_name]
    score = getattr(lib, 'validation')(algo_name, database, **opt)
    return score


def accuracy(algo_name, database, libs=['lightfm']):
    assert algo_name in ['als', 'bpr', 'warp']
    assert database in ['kakao_reco_730m', 'ml20m', 'ml100k', 'kakao_brunch_12m']
    L = {'buffalo': BuffaloLib,
         'lightfm': LightfmLib}
    rets = {}
    for lib in libs:
        rets[lib] = _get_validation_score(algo_name, L[lib](), database)
    _print_table(rets)


if __name__ == '__main__':
    fire.Fire({'compare_warp_bpr': compare_warp_bpr,
               'accuracy': accuracy})
