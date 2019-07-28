# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
# -*- coding: utf-8 -*-
import time
import json
import logging

import tqdm
import cython
from libcpp cimport bool
from libc.stdint cimport int64_t
from libcpp.string cimport string
from eigency.core cimport MatrixXf, Map, VectorXi, VectorXf

import numpy as np
cimport numpy as np
from numpy.linalg import norm
from hyperopt import STATUS_OK as HOPT_STATUS_OK

import buffalo.data
from buffalo.data.base import Data
from buffalo.misc import aux, log
from buffalo.evaluate import Evaluable
from buffalo.algo.options import BprmfOption
from buffalo.algo.optimize import Optimizable
from buffalo.data.buffered_data import BufferedDataMatrix
from buffalo.algo.base import Algo, Serializable, TensorboardExtention


cdef extern from "buffalo/algo_impl/bpr/bpr.hpp" namespace "bpr":
    cdef cppclass CBPRMF:
        void release() nogil except +
        bool init(string) nogil except +
        void set_factors(Map[MatrixXf]&,
                         Map[MatrixXf]&,
                         Map[MatrixXf]&) nogil except +
        void precompute(int) nogil except +
        double partial_update(int,
                              int,
                              int64_t*,
                              Map[VectorXi]&,
                              Map[VectorXi]&) nogil except +
        double compute_loss(Map[VectorXi]&,
                            Map[VectorXi]&,
                            Map[VectorXi]&) nogil except +
        double update_parameters()


cdef class PyBPRMF:
    """CBPRMF object holder"""
    cdef CBPRMF* obj  # C-BPRMF object

    def __cinit__(self):
        self.obj = new CBPRMF()

    def __dealloc__(self):
        self.obj.release()
        del self.obj

    def init(self, option_path):
        return self.obj.init(option_path)

    def set_factors(self,
                    np.ndarray[np.float32_t, ndim=2] P,
                    np.ndarray[np.float32_t, ndim=2] Q,
                    np.ndarray[np.float32_t, ndim=2] Qb):
        self.obj.set_factors(Map[MatrixXf](P),
                             Map[MatrixXf](Q),
                             Map[MatrixXf](Qb))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def partial_update(self,
                       int start_x,
                       int next_x,
                       np.ndarray[np.int64_t, ndim=1] indptr,
                       np.ndarray[np.int32_t, ndim=1] positives,
                       np.ndarray[np.int32_t, ndim=1] negatives):
        return self.obj.partial_update(start_x,
                                       next_x,
                                       &indptr[0],
                                       Map[VectorXi](positives),
                                       Map[VectorXi](negatives))

    def compute_loss(self,
                     np.ndarray[np.int32_t, ndim=1] users,
                     np.ndarray[np.int32_t, ndim=1] positives,
                     np.ndarray[np.int32_t, ndim=1] negatives):
        return self.obj.compute_loss(Map[VectorXi](users),
                                     Map[VectorXi](positives),
                                     Map[VectorXi](negatives))

    def update_parameters(self):
        return self.obj.update_parameters()


class BPRMF(Algo, BprmfOption, Evaluable, Serializable, Optimizable, TensorboardExtention):
    """Python implementation for C-BPRMF.
    """
    def __init__(self, opt_path, *args, **kwargs):
        Algo.__init__(self, *args, **kwargs)
        BprmfOption.__init__(self, *args, **kwargs)
        Evaluable.__init__(self, *args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        Optimizable.__init__(self, *args, **kwargs)

        self.logger = log.get_logger('BPRMF')
        self.opt, self.opt_path = self.get_option(opt_path)
        self.obj = PyBPRMF()
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

    def set_data(self, data):
        assert isinstance(data, aux.data.Data), 'Wrong instance: {}'.format(type(data))
        self.data = data

    def fast_similar(self, onoff=True):
        # TODO: implement
        if onoff:
            self.FQ = self.normalize(self.Q)
        else:
            if hasattr(self, 'FQ'):
                del self.FQ

    def initialize(self):
        assert self.data, 'Data is not setted'
        if self.opt.random_seed:
            np.random.seed(self.opt.random_seed)
        self.buf = BufferedDataMatrix()
        self.buf.initialize(self.data)
        self.init_factors()
        self.prepare_sampling()

    def init_factors(self):
        header = self.data.get_header()
        self.P = np.abs(np.random.normal(scale=1.0/(self.opt.d ** 2),
                                         size=(header['num_users'], self.opt.d)).astype("float32"), order='F')
        self.Q = np.abs(np.random.normal(scale=1.0/(self.opt.d ** 2),
                                         size=(header['num_items'], self.opt.d)).astype("float32"), order='F')
        self.Qb = np.abs(np.random.normal(scale=1.0/(self.opt.d ** 2),
                                          size=(header['num_items'], 1)).astype("float32"), order='F')
        self.obj.set_factors(self.P, self.Q, self.Qb)

    def prepare_sampling(self):
        self.logger.info('Preparing sampling ...')
        header = self.data.get_header()
        if self.opt.sampling_power > 0.0:
            self.sampling_weights_ = np.zeros(header['num_items'], dtype=np.float32)
            for sz in self.buf.fetch_batch():
                start_x, next_x, indptr, keys, vals = self.buf.get()
                for key in keys:
                    self.sampling_weights_[key] += 1.0
            self.sampling_weights_ = self.sampling_weights_ ** self.opt.sampling_power
            self.sampling_weights_ /= np.sum(self.sampling_weights_)
            self.sampling_method = 'with_p'
        else:
            self.sampling_method = 'uniform'

    def _get_topk_recommendation(self, rows, topk):
        p = self.P[rows]
        topks = np.argsort(p.dot(self.Q.T) + self.Qb.T, axis=1)[:, -topk:][:,::-1]
        return zip(rows, topks)

    def _get_most_similar(self, cols, topk):
        if hasattr(self, 'FQ'):
            q = self.FQ[cols]
            topks = np.argsort(q.dot(self.FQ.T), axis=1)[:, -topk:][:,::-1]
        else:
            q = self.Q[cols]
            dot = q.dot(self.Q.T)
            dot = dot / (norm(q) * norm(self.Q, axis=1))
            topks = np.argsort(dot, axis=1)[:, -topk:][:,::-1]
        return zip(cols, topks)

    def get_scores(self, row_col_pairs):
        rets = {(r, c): self.P[r].dot(self.Q[c]) + self.Qb[c][0] for r, c in row_col_pairs}
        return rets

    def sampling_loss_samples(self):
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
        self._sub_samples = [
            np.array(users, dtype=np.int32, order='F'),
            np.array(positives, dtype=np.int32, order='F'),
            np.array(negatives, dtype=np.int32, order='F')
        ]
        self.logger.info('Generated %s loss samples.' % len(users))

    def generate_negative_samples(self, start_x, next_x, indptr, keys, vals):
        end_loop = next_x - start_x
        total_positives = keys.shape[0]
        num_negative_samples = self.opt.num_negative_samples
        total_negatives = total_positives * num_negative_samples
        negative_samples = np.zeros(shape=total_negatives, dtype=np.int32, order='F')
        negative_sample_index = 0
        shifted = 0 if start_x == 0 else indptr[start_x - 1]
        for index in range(end_loop):
            u = start_x + index
            beg = 0 if u == 0 else indptr[u - 1]
            end = indptr[u]
            data_size = end - beg
            seen = set(keys[shifted + beg: shifted + end])
            neg_data_size = data_size * num_negative_samples
            # TODO: deal with self.opt.negative_sampling_with_replacement
            if self.sampling_method == 'with_p':
                negs = np.random.choice(range(self.Q.shape[0]),
                                        size=neg_data_size + data_size,
                                        replace=False,
                                        p=self.sampling_weights_)
            else:  # uniform
                negs = np.random.choice(range(self.Q.shape[0]),
                                        size=neg_data_size + data_size,
                                        replace=False)
            negs = [n for n in negs if n not in seen]
            negative_samples[negative_sample_index:negative_sample_index + neg_data_size] = negs[:neg_data_size]
            negative_sample_index += neg_data_size
        return negative_samples

    def _iterate(self):
        header = self.data.get_header()
        end = header['num_users']
        update_t, feed_t, sampling_t, updated = 0, 0, 0, 0
        self.buf.set_group('rowwise')
        with log.pbar(log.DEBUG,
                      total=header['num_nnz'], mininterval=30) as pbar:
            start_t = time.time()
            for sz in self.buf.fetch_batch():
                updated += sz
                feed_t += time.time() - start_t
                start_x, next_x, indptr, keys, vals = self.buf.get()

                start_t = time.time()
                negs = self.generate_negative_samples(start_x, next_x, indptr, keys, vals)
                sampling_t += time.time() - start_t

                start_t = time.time()
                self.obj.partial_update(start_x, next_x, indptr, keys, negs)
                update_t += time.time() - start_t
                pbar.update(sz)
            pbar.refresh()
        cplex = self.obj.update_parameters()
        self.logger.debug(f'updated processed({updated}) elapsed(data feed: {feed_t:0.3f} update: {update_t:0.3f} sample: {sampling_t:0.3f})')
        return cplex

    def compute_loss(self):
        return self.obj.compute_loss(self._sub_samples[0],
                                     self._sub_samples[1],
                                     self._sub_samples[2])

    def train(self):
        rmse, self.validation_result = None, {}
        self.initialize_tensorboard(self.opt.num_iters)
        self.sampling_loss_samples()
        for i in range(self.opt.num_iters):
            start_t = time.time()
            cplex = self._iterate()
            loss = self.compute_loss()
            self.logger.info('Iteration %s: ModelComplexity %.3f LOSS %.3f Elapsed %.3f secs' % (i + 1, cplex, loss, time.time() - start_t))
            metrics = {'auc': loss}
            if self.opt.validation:
                self.validation_result = self.get_validation_results()
                self.logger.info('Validation: ' + \
                                 ' '.join([f'{k}:{v:0.5f}'
                                           for k, v in self.validation_result.items()]))
                metrics.update({'val_%s' % k: v
                                for k, v in self.validation_result.items()})
            self.update_tensorboard_data(metrics)
        ret = {'auc': loss}
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
        return [('opt', self.opt),
                ('Q', self.Q),
                ('Qb', self.Qb),
                ('P', self.P)]

    def get_evaluation_metrics(self):
        return ['rmse', 'val_rmse', 'val_ndcg', 'val_map', 'val_accuracy', 'val_error']
