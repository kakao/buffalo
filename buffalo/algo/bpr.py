# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
# -*- coding: utf-8 -*-
import time
import json

import numpy as np
from numpy.linalg import norm
from hyperopt import STATUS_OK as HOPT_STATUS_OK

import buffalo.data
from buffalo.misc import aux, log
from buffalo.data.base import Data
from buffalo.algo._bpr import CyBPRMF
from buffalo.evaluate import Evaluable
from buffalo.algo.options import BPRMFOption
from buffalo.algo.optimize import Optimizable
from buffalo.data.buffered_data import BufferedDataMatrix
from buffalo.algo.base import Algo, Serializable, TensorboardExtention


class BPRMF(Algo, BPRMFOption, Evaluable, Serializable, Optimizable, TensorboardExtention):
    """Python implementation for C-BPRMF.
    """
    def __init__(self, opt_path=None, *args, **kwargs):
        Algo.__init__(self, *args, **kwargs)
        BPRMFOption.__init__(self, *args, **kwargs)
        Evaluable.__init__(self, *args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        Optimizable.__init__(self, *args, **kwargs)
        if opt_path is None:
            opt_path = BPRMFOption().get_default_option()

        self.logger = log.get_logger('BPRMF')
        self.opt, self.opt_path = self.get_option(opt_path)
        self.obj = CyBPRMF()
        assert self.obj.init(bytes(self.opt_path, 'utf-8')),\
            'cannot parse option file: %s' % opt_path
        self.data = None
        data = kwargs.get('data')
        data_opt = self.opt.get('data_opt')
        data_opt = kwargs.get('data_opt', data_opt)
        if data_opt:
            self.data = buffalo.data.load(data_opt)
            self.data.create()
        elif isinstance(data, Data):
            self.data = data
        self.logger.info('BPRMF(%s)' % json.dumps(self.opt, indent=2))
        if self.data:
            self.logger.info(self.data.show_info())
            assert self.data.data_type in ['matrix']

    @staticmethod
    def new(path, data_fields=[]):
        return BPRMF.instantiate(BPRMFOption, path, data_fields)

    def set_data(self, data):
        assert isinstance(data, aux.data.Data), 'Wrong instance: {}'.format(type(data))
        self.data = data

    def normalize(self, group='item'):
        if group == 'item' and not self.opt._nrz_Q:
            self.Q = self._normalize(self.Q)
            self.opt._nrz_Q = True
        elif group == 'user' and not self.opt._nrz_P:
            self.P = self._normalize(self.P)
            self.opt._nrz_P = True

    def initialize(self):
        super().initialize()
        assert self.data, 'Data is not setted'
        if self.opt.random_seed:
            np.random.seed(self.opt.random_seed)
        self.buf = BufferedDataMatrix()
        self.buf.initialize(self.data)
        self.init_factors()
        self.prepare_sampling()

    def init_factors(self):
        header = self.data.get_header()
        for attr_name in ['P', 'Q', 'Qb']:
            setattr(self, attr_name, None)
        self.P = np.abs(np.random.normal(scale=1.0/(self.opt.d ** 2),
                                         size=(header['num_users'], self.opt.d)).astype("float32"), order='C')
        self.Q = np.abs(np.random.normal(scale=1.0/(self.opt.d ** 2),
                                         size=(header['num_items'], self.opt.d)).astype("float32"), order='C')
        self.Qb = np.abs(np.random.normal(scale=1.0/(self.opt.d ** 2),
                                          size=(header['num_items'], 1)).astype("float32"), order='C')
        self.obj.initialize_model(self.P, self.Q, self.Qb, header['num_nnz'])

    def prepare_sampling(self):
        self.logger.info('Preparing sampling ...')
        header = self.data.get_header()
        self.sampling_table_ = np.zeros(header['num_items'], dtype=np.int64)
        if self.opt.sampling_power > 0.0:
            for sz in self.buf.fetch_batch():
                *_, keys, vals = self.buf.get()
                for i in range(sz):
                    self.sampling_table_[keys[i]] += 1
            self.sampling_table_ **= int(self.opt.sampling_power)
            for i in range(1, header['num_items']):
                self.sampling_table_[i] += self.sampling_table_[i - 1]
        self.obj.set_cumulative_table(self.sampling_table_, header['num_items'])

    def _get_topk_recommendation(self, rows, topk, pool=None):
        p = self.P[rows]
        topks = super()._get_topk_recommendation(p, self.Q, pool, topk, self.opt.num_workers)
        return zip(rows, topks)

    def _get_most_similar_item(self, col, topk, pool):
        return self._get_most_similar(col, topk, self.Q, self.opt._nrz_Q, pool)

    def get_scores(self, row_col_pairs):
        rets = {(r, c): self.P[r].dot(self.Q[c]) + self.Qb[c][0] for r, c in row_col_pairs}
        return rets

    def sampling_loss_samples(self):
        users, positives, negatives = [], [], []
        if self.opt.compute_loss_on_training:
            self.logger.info('Sampling loss samples...')
            header = self.data.get_header()
            num_loss_samples = int(header['num_users'] ** 0.5)
            _users = np.random.choice(range(self.P.shape[0]), size=num_loss_samples, replace=False)
            users = []
            positives, negatives = [], []
            for u in _users:
                keys, *_ = self.data.get(u)
                if len(keys) == 0:
                    continue
                seen = set(keys)
                negs = np.random.choice(range(self.Q.shape[0]),
                                        size=len(seen) + 1,
                                        replace=False)
                negs = [n for n in negs if n not in seen]
                users.append(u)
                positives.append(keys[0])
                negatives.append(negs[0])
            self.logger.info('Generated %s loss samples.' % len(users))
        self._sub_samples = [
            np.array(users, dtype=np.int32, order='F'),
            np.array(positives, dtype=np.int32, order='F'),
            np.array(negatives, dtype=np.int32, order='F')
        ]

    def _get_feature(self, index, group='item'):
        if group == 'item':
            return self.Q[index]
        elif group == 'user':
            return self.P[index]
        return None

    def _iterate(self):
        header = self.data.get_header()
        end = header['num_users']
        update_t, feed_t, updated = 0, 0, 0
        self.buf.set_group('rowwise')
        with log.ProgressBar(log.DEBUG,
                             total=header['num_nnz'], mininterval=30) as pbar:
            start_t = time.time()
            for sz in self.buf.fetch_batch():
                updated += sz
                feed_t += time.time() - start_t
                start_x, next_x, indptr, keys, vals = self.buf.get()

                start_t = time.time()
                self.obj.add_jobs(start_x, next_x, indptr, keys)
                update_t += time.time() - start_t
                pbar.update(sz)
            pbar.refresh()
        self.obj.update_parameters()
        self.logger.debug(f'updated processed({updated}) elapsed(data feed: {feed_t:0.3f} update: {update_t:0.3f}')

    def compute_loss(self):
        return self.obj.compute_loss(self._sub_samples[0],
                                     self._sub_samples[1],
                                     self._sub_samples[2])

    def train(self):
        rmse, self.validation_result = None, {}
        self.prepare_evaluation()
        self.initialize_tensorboard(self.opt.num_iters)
        self.sampling_loss_samples()
        self.obj.launch_workers()
        best_loss = 987654321.0
        for i in range(self.opt.num_iters):
            start_t = time.time()
            self._iterate()
            self.obj.wait_until_done()
            loss = self.compute_loss() if self.opt.compute_loss_on_training else 0.0
            train_t = time.time() - start_t

            metrics = {'train_loss': loss}
            if self.opt.validation and self.opt.evaluation_on_learning and self.periodical(self.opt.evaluation_period, i):
                start_t = time.time()
                self.validation_result = self.get_validation_results()
                vali_t = time.time() - start_t
                val_str = ' '.join([f'{k}:{v:0.5f}' for k, v in self.validation_result.items()])
                self.logger.info(f'Validation: {val_str} Elased {vali_t:0.3f}')
                metrics.update({'val_%s' % k: v
                                for k, v in self.validation_result.items()})
            self.logger.info('Iteration %s: PR-Loss %.3f Elapsed %.3f secs' % (i + 1, loss, time.time() - start_t))
            self.update_tensorboard_data(metrics)
            best_loss = self.save_best_only(loss, best_loss, i)
            if self.early_stopping(loss):
                break
        loss = self.obj.join()

        ret = {'train_loss': loss}
        ret.update({'val_%s' % k: v
                    for k, v in self.validation_result.items()})
        self.finalize_tensorboard()
        return ret

    def _optimize(self, params):
        # TODO: implement
        self._optimize_params = params
        for name, value in params.items():
            assert name in self.opt, 'Unexepcted parameter: {}'.format(name)
            setattr(self.opt, name, value)
        with open(self._temporary_opt_file, 'w') as fout:
            json.dump(self.opt, fout, indent=2)
        assert self.obj.init(bytes(self._temporary_opt_file, 'utf-8')),\
            'cannot parse option file: %s' % self._temporary_opt_file
        self.logger.info(params)
        self.init_factors()
        loss = self.train()
        loss['eval_time'] = time.time()
        loss['loss'] = loss.get(self.opt.optimize.loss)
        # TODO: deal with failture of training
        loss['status'] = HOPT_STATUS_OK
        self._optimize_loss = loss
        return loss

    def _get_data(self):
        data = super()._get_data()
        data.extend([('opt', self.opt),
                     ('Q', self.Q),
                     ('Qb', self.Qb),
                     ('P', self.P)])
        return data

    def get_evaluation_metrics(self):
        return ['val_rmse', 'val_ndcg', 'val_map', 'val_accuracy', 'val_error', 'train_loss']
