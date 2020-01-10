# cython: experimental_cpp_class_def=True, language_level=3
# distutils: language=c++
# -*- coding: utf-8 -*-
import time
import json

import numpy as np
from hyperopt import STATUS_OK as HOPT_STATUS_OK

import buffalo.data
from buffalo.misc import aux, log
from buffalo.data.base import Data
from buffalo.algo._w2v import CyW2V
from buffalo.evaluate import Evaluable
from buffalo.algo.options import W2VOption
from buffalo.algo.optimize import Optimizable
from buffalo.data.buffered_data import BufferedDataStream
from buffalo.algo.base import Algo, Serializable, TensorboardExtention


class W2V(Algo, W2VOption, Evaluable, Serializable, Optimizable, TensorboardExtention):
    """Python implementation for C-W2V
    """
    def __init__(self, opt_path=None, *args, **kwargs):
        Algo.__init__(self, *args, **kwargs)
        W2VOption.__init__(self, *args, **kwargs)
        Evaluable.__init__(self, *args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        Optimizable.__init__(self, *args, **kwargs)
        if opt_path is None:
            opt_path = W2VOption().get_default_option()

        self.logger = log.get_logger('W2V')
        self.opt, self.opt_path = self.get_option(opt_path)
        self.obj = CyW2V()
        assert self.obj.init(bytes(self.opt_path, 'utf-8')), 'cannot parse option file: %s' % opt_path
        self.data = None
        data = kwargs.get('data')
        data_opt = self.opt.get('data_opt')
        data_opt = kwargs.get('data_opt', data_opt)
        if data_opt:
            self.data = buffalo.data.load(data_opt)
            assert self.data.data_type == 'stream'
            self.data.create()
        elif isinstance(data, Data):
            self.data = data
        self.logger.info('W2V(%s)' % json.dumps(self.opt, indent=2))
        if self.data:
            self.logger.info(self.data.show_info())
            assert self.data.data_type in ['stream']
        self._vocab = aux.Option({'size': 0,
                                  'index': None,
                                  'inv_index': None,
                                  'scale': None,
                                  'dist': None,
                                  'total_word_count': 0})

    @staticmethod
    def new(path, data_fields=[]):
        return W2V.instantiate(W2VOption, path, data_fields)

    def set_data(self, data):
        assert isinstance(data, aux.data.Data), 'Wrong instance: {}'.format(type(data))
        self.data = data

    def normalize(self, group='item'):
        if group == 'item' and not self.opt._nrz_L0:
            self.L0 = self._normalize(self.L0)
            self.opt._nrz_L0 = True

    def get_index(self, key, group='item'):
        is_many = isinstance(key, list)
        indexes = super().get_index(key, group)
        if not is_many:
            indexes = [indexes]
        indexes = [i if i is None or self._vocab.index[i] < 1 else self._vocab.index[i] - 1
                   for i in indexes]
        if not is_many:
            return indexes[0]
        return np.array(indexes)

    def _get_feature(self, index, group='item'):
        if group == 'item':
            return self.L0[index]
        return None

    def initialize(self):
        super().initialize()
        assert self.data, 'Data is not setted'
        self.buf = BufferedDataStream()
        self.buf.initialize(self.data)
        self.build_vocab()
        self.init_factors(self._vocab.size)
        self.obj.initialize_model(self.L0, self._vocab.index, self._vocab.scale,
                                  self._vocab.dist, self._vocab.total_word_count)

    def build_vocab(self):
        header = self.data.get_header()
        self.logger.info('Caculating the frequency of words.')
        uni = [0 for i in range(header['num_items'])]
        total_word_count = 0
        for sz in self.buf.fetch_batch():
            *_, keys = self.buf.get()
            for i in range(sz):
                uni[keys[i]] += 1
            total_word_count += sz
        self.logger.info(f'Reducing vocab with min_count({self.opt.min_count}).')
        total_vocab = 0
        # use = np.zeros(shape=header['num_items'], dtype=np.int32, order='F')
        use = [0 for i in range(header['num_items'])]
        for i in range(header['num_items']):
            if uni[i] >= self.opt.min_count:
                total_vocab += 1
                use[i] = total_vocab
        self.logger.info(f'Scaling vocab({total_vocab}) with sample({self.opt.sample}).')
        scale = np.zeros(shape=total_vocab, dtype=np.uint32, order='C')
        threshold_count = sum([uni[i] for i in range(header['num_items']) if use[i]])
        if self.opt.sample > 0.0:
            threshold_count *= self.opt.sample
        num_downsampled = 0
        for i in range(header['num_items']):
            if not use[i]:
                continue
            p = (((uni[i] / threshold_count) ** 0.5) + 1) * (threshold_count / uni[i])
            if p < 1.0:
                num_downsampled += 1
            else:
                p = 1.0
            scale[use[i] - 1] = p * 0xFFFFFFFF
        self.logger.info(f'Downsampled {num_downsampled} most-common words.')
        dist = self.get_sampling_distribution(uni, use, total_vocab)
        self._vocab.size = total_vocab
        self._vocab.scale = scale
        self._vocab.index = np.array(use, dtype=np.int32, order='C')
        self._vocab.inv_index = np.array([idx for idx, u in enumerate(use) if u > 0],
                                         dtype=np.int32)
        self._vocab.dist = dist
        self._vocab.total_word_count = total_word_count
        self.logger.info(f'Vocab({total_vocab}) TotalWords({total_word_count})')

    def init_factors(self, vocab_size):
        self.L0 = None
        self.L0 = np.abs(np.random.normal(scale=1.0 / (self.opt.d ** 2),
                                          size=(vocab_size, self.opt.d)).astype("float32"))

    def get_sampling_distribution(self, uni, use, total_vocab):
        dist0 = np.zeros(shape=total_vocab, dtype=np.float64, order='C')
        for i in range(len(use)):
            if not use[i]:
                continue
            dist0[use[i] - 1] = uni[i]
        dist0 = dist0 ** 0.75
        train_word_pow = dist0.sum()
        dist0 /= train_word_pow

        dist = np.zeros(shape=total_vocab, dtype=np.int32, order='C')
        summed = 0.0
        domain = 0x7FFFFFFF
        for i in range(total_vocab):
            summed += dist0[i]
            dist[i] = summed * domain
        assert abs(dist[-1] - domain) < 3  # small difference is okay
        return dist

    def _get_topk_recommendation(self, rows, topk, pool=None):
        raise NotImplementedError

    def _get_most_similar_item(self, col, topk, pool):
        if not isinstance(col, np.ndarray):
            col = self._vocab.index[col] - 1
            if col < 0:
                return [], []
        topks, scores = super()._get_most_similar_item(col, topk, self.L0, self.opt._nrz_L0, pool)
        topks = self._vocab.inv_index[topks]
        return topks, scores

    def get_scores(self, row_col_pairs):
        return []

    def _iterate(self):
        header = self.data.get_header()
        # end = header['num_users']
        update_t, feed_t, updated = 0, 0, 0
        self.buf.set_group('rowwise')
        with log.ProgressBar(log.DEBUG,
                             total=header['num_nnz'], mininterval=15) as pbar:
            start_t = time.time()
            for sz in self.buf.fetch_batch():
                updated += sz
                feed_t += time.time() - start_t
                start_x, next_x, indptr, keys = self.buf.get()

                start_t = time.time()
                self.obj.add_jobs(start_x, next_x, indptr, keys)
                update_t += time.time() - start_t
                pbar.update(sz)
        self.logger.debug(f'processed({updated}) elapsed(data feed: {feed_t:0.3f}s update: {update_t:0.3f})s')

    def train(self):
        self.validation_result = {}
        self.obj.launch_workers()
        for i in range(self.opt.num_iters):
            start_t = time.time()
            self._iterate()
            self.logger.info('Iteration %s: Elapsed %.3f secs' % (i + 1, time.time() - start_t))
        # loss = self.obj.join()
        self.obj.join()
        return {}

    def _optimize(self, params):
        # TODO: implement
        self._optimize_params = params
        for name, value in params.items():
            assert name in self.opt, 'Unexepcted parameter: {}'.format(name)
            if isinstance(value, np.generic):
                setattr(self.opt, name, value.item())
            else:
                setattr(self.opt, name, value)
        with open(self._temporary_opt_file, 'w') as fout:
            json.dump(self.opt, fout, indent=2)
        assert self.obj.init(bytes(self._temporary_opt_file, 'utf-8')),\
            'cannot parse option file: %s' % self._temporary_opt_file
        self.logger.info(params)
        self.init_factors()
        loss = self.train()
        loss['loss'] = loss.get(self.opt.optimize.loss)
        # TODO: deal with failture of training
        loss['status'] = HOPT_STATUS_OK
        self._optimize_loss = loss
        return loss

    def _get_data(self):
        data = super()._get_data()
        data.extend([('opt', self.opt),
                     ('L0', self.L0),
                     ('_vocab', self._vocab)])
        return data

    def get_evaluation_metrics(self):
        return []
