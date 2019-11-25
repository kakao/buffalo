# -*- coding: utf-8 -*-
import json
import time

import fire

from buffalo.algo.als import ALS
from buffalo.misc import aux, log
from buffalo.parallel.base import ParALS
from buffalo.algo.options import ALSOption
from buffalo.data.mm import MatrixMarketOptions


def example1():
    log.set_log_level(log.DEBUG)
    als_option = ALSOption().get_default_option()
    als_option.validation = aux.Option({'topk': 10})
    data_option = MatrixMarketOptions().get_default_option()
    data_option.input.main = '../tests/ext/ml-100k/main'
    data_option.input.iid = '../tests/ext/ml-100k/iid'

    als = ALS(als_option, data_opt=data_option)
    als.initialize()
    als.train()
    print('MovieLens 100k metrics for validations\n%s' % json.dumps(als.get_validation_results(), indent=2))

    print('Similar movies to Star_Wars_(1977)')
    for rank, (movie_name, score) in enumerate(als.most_similar('49.Star_Wars_(1977)')):
        print(f'{rank + 1:02d}. {score:.3f} {movie_name}')

    print('Run hyper parameter optimization for val_ndcg...')
    als.opt.num_workers = 4
    als.opt.evaluation_period = 10
    als.opt.optimize = aux.Option({
        'loss': 'val_ndcg',
        'max_trials': 100,
        'deployment': True,
        'start_with_default_parameters': True,
        'space': {
            'd': ['randint', ['d', 10, 128]],
            'reg_u': ['uniform', ['reg_u', 0.1, 1.0]],
            'reg_i': ['uniform', ['reg_i', 0.1, 1.0]],
            'alpha': ['randint', ['alpha', 1, 10]],
        }
    })
    log.set_log_level(log.INFO)
    als.opt.model_path = './example1.ml100k.als.optimize.bin'
    print(json.dumps({'alpha': als.opt.alpha, 'd': als.opt.d,
                      'reg_u': als.opt.reg_u, 'reg_i': als.opt.reg_i}, indent=2))
    als.optimize()
    als.load('./example1.ml100k.als.optimize.bin')

    print('Similar movies to Star_Wars_(1977)')
    for rank, (movie_name, score) in enumerate(als.most_similar('49.Star_Wars_(1977)')):
        print(f'{rank + 1:02d}. {score:.3f} {movie_name}')

    optimization_res = als.get_optimization_data()
    best_parameters = optimization_res['best_parameters']

    print(json.dumps(optimization_res['best'], indent=2))
    print(json.dumps({'alpha': best_parameters['alpha'], 'd': best_parameters['d'],
                      'reg_u': best_parameters['reg_u'], 'reg_i': best_parameters['reg_i']}, indent=2))


def example2():
    log.set_log_level(log.INFO)
    als_option = ALSOption().get_default_option()
    data_option = MatrixMarketOptions().get_default_option()
    data_option.input.main = '../tests/ext/ml-20m/main'
    data_option.input.iid = '../tests/ext/ml-20m/iid'
    data_option.data.path = './ml20m.h5py'
    data_option.data.use_cache = True

    als = ALS(als_option, data_opt=data_option)
    als.initialize()
    als.train()
    als.normalize('item')
    als.build_itemid_map()

    print('Make item recommendation on als.ml20m.par.top10.tsv with Paralell(Thread=4)')
    par = ParALS(als)
    par.num_workers=4
    all_items = als._idmanager.itemids
    start_t = time.time()
    with open('als.ml20m.par.top10.tsv', 'w') as fout:
        for idx in range(0, len(all_items), 128):
            topks, _ = par.most_similar(all_items[idx:idx + 128], repr=True)
            for q, p in zip(all_items[idx:idx + 128], topks):
                fout.write('%s\t%s\n' % (q, '\t'.join(p)))
    print('took: %.3f secs' % (time.time() - start_t))

    from n2 import HnswIndex
    index = HnswIndex(als.Q.shape[1])
    for f in als.Q:
        index.add_data(f)
    index.build(n_threads=4)
    index.save('ml20m.n2.index')
    index.unload()
    print('Make item recommendation on als.ml20m.par.top10.tsv with Ann(Thread=1)')
    par.set_hnsw_index('ml20m.n2.index', 'item')
    par.num_workers=4
    start_t = time.time()
    with open('als.ml20m.ann.top10.tsv', 'w') as fout:
        for idx in range(0, len(all_items), 128):
            topks, _ = par.most_similar(all_items[idx:idx + 128], repr=True)
            for q, p in zip(all_items[idx:idx + 128], topks):
                fout.write('%s\t%s\n' % (q, '\t'.join(p)))
    print('took: %.3f secs' % (time.time() - start_t))

if __name__ == '__main__':
    fire.Fire({'example1': example1,
               'example2': example2})
