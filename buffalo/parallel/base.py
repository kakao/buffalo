# -*- coding: utf-8 -*-
import abc

import numpy as np

from buffalo.algo.als import ALS
from buffalo.algo.cfr import CFR
from buffalo.algo.w2v import W2V
from buffalo.algo.bpr import BPRMF

from buffalo.parallel._core import dot_topn, ann_search


class Parallel(abc.ABC):
    def __init__(self, algo, *argv, **kwargs):
        super().__init__()
        if not isinstance(algo, (ALS, CFR, W2V, BPRMF)):
            raise ValueError('Not supported algo type: %s' % type(algo))
        self.algo = algo
        self.num_workers = int(kwargs['num_workers'])
        self._ann_list = {}

    def _most_similar(self, group, indexes, Factor, topk, pool, ef_search, use_mmap):
        dummy_bias = np.array([[]], dtype=np.float32)
        out_keys = np.zeros(shape=(len(indexes), topk), dtype=np.int32)
        out_scores = np.zeros(shape=(len(indexes), topk), dtype=np.float32)
        if group in self._ann_list:
            if ef_search == -1:
                ef_search = topk * 10
            ann_search(self._ann_list[group].encode('utf8'), ef_search, use_mmap, indexes, Factor, Factor, dummy_bias, out_keys, out_scores, pool, topk, self.num_workers)
        else:
            dot_topn(indexes, Factor, Factor, dummy_bias, out_keys, out_scores, pool, topk, self.num_workers)
        return out_keys, out_scores

    def set_hnsw_index(self, path, group):
        """ Set N2 HNSW Index for apporximate nearest neighbor features.

        N2 is a open source project for approximate nearest neighbor.
        For more detail please see the https://github.com/kakao/n2.

        :param str path: The path of hnsw index file.
        :param str group: Indexed data group.
        """
        self._ann_list[group] = path

    @abc.abstractmethod
    def most_similar(self, keys, topk=10, group='item', pool=None, repr=False, ef_search=-1, use_mmap=True):
        """Caculate TopK most similar items for each keys in parallel processing.

        :param list keys: Query Keys
        :param int topk: Number of topK
        :param str group: Data group where to find (default: item)
        :param pool: The list of item keys to find for.
            If it is a numpy.ndarray instance then it treat as index of items and it would be helpful for calculation speed. (default: None)
        :type pool: list or numpy.ndarray
        :param bool repr: Set True, to return as item key instead index.
        :param int ef_search: This parameter is passed to N2 when hnsw_index was given for the group. (default: -1 which means topk * 10)
        :param use_mmap: This parameter is passed to N2 when hnsw_index given for the group. (default: True)
        :return: list of tuple(key, score)
        """
        raise NotImplementedError

    def _topk_recommendation(self, indexes, FactorP, FactorQ, topk, pool):
        dummy_bias = np.array([[]], dtype=np.float32)
        out_keys = np.zeros(shape=(len(indexes), topk), dtype=np.int32)
        out_scores = np.zeros(shape=(len(indexes), topk), dtype=np.float32)
        dot_topn(indexes, FactorP, FactorQ, dummy_bias, out_keys, out_scores, pool, topk, self.num_workers)
        return out_keys, out_scores

    def _topk_recommendation_bias(self, indexes, FactorP, FactorQ, FactorQb, topk, pool):
        out_keys = np.zeros(shape=(len(indexes), topk), dtype=np.int32)
        out_scores = np.zeros(shape=(len(indexes), topk), dtype=np.float32)
        dot_topn(indexes, FactorP, FactorQ, FactorQb, out_keys, out_scores, pool, topk, self.num_workers)
        return out_keys, out_scores

    @abc.abstractmethod
    def topk_recommendation(self, keys, topk=10, pool=None, repr=False):
        """Caculate TopK recommendation for each users in parallel processing.

        :param list keys: Query Keys
        :param int topk: Number of topK
        :param bool repr: Set True, to return as item key instead index.
        :return: list of tuple(key, score)
        """
        raise NotImplementedError


class ParALS(Parallel):
    def __init__(self, algo, **kwargs):
        num_workers = int(kwargs.get('num_workers', algo.opt.num_workers))
        super().__init__(algo, num_workers=num_workers)

    def most_similar(self, keys, topk=10, group='item', pool=None, repr=False, ef_search=-1, use_mmap=True):
        """See the documentation of Parallel."""
        self.algo.normalize(group=group)
        indexes = self.algo.get_index_pool(keys, group=group)
        keys = [k for k, i in zip(keys, indexes) if i is not None]
        indexes = np.array([i for i in indexes if i is not None], dtype=np.int32)
        if pool is not None:
            pool = self.algo.get_index_pool(pool, group=group)
            if len(pool) == 0:
                raise RuntimeError('pool is empty')
        else:
            # It assume that empty pool menas for all items
            pool = np.array([], dtype=np.int32)
        if group == 'item':
            topks, scores = super()._most_similar(group, indexes, self.algo.Q, topk, pool, ef_search, use_mmap)
            if repr:
                topks = [[self.algo._idmanager.itemids[t] for t in tt if t != -1] for tt in topks]
            return topks, scores
        elif group == 'user':
            topks, scores = super()._most_similar(group, indexes, self.algo.P, topk, pool, ef_search, use_mmap)
            if repr:
                topks = [[self.algo._idmanager.userids[t] for t in tt if t != -1] for tt in topks]
            return topks, scores
        raise ValueError(f'Not supported group: {group}')

    def topk_recommendation(self, keys, topk=10, pool=None, repr=False):
        """See the documentation of Parallel."""
        if self.algo.opt._nrz_P or self.algo.opt._nrz_Q:
            raise RuntimeError('Cannot make topk recommendation with normalized factors')
        # It is possible to skip make recommendation for not-existed keys.
        indexes = self.algo.get_index_pool(keys, group='user')
        keys = [k for k, i in zip(keys, indexes) if i is not None]
        indexes = np.array([i for i in indexes if i is not None], dtype=np.int32)
        if pool is not None:
            pool = self.algo.get_index_pool(pool, group='item')
            if len(pool) == 0:
                raise RuntimeError('pool is empty')
        else:
            # It assume that empty pool menas for all items
            pool = np.array([], dtype=np.int32)
        topks, scores = super()._topk_recommendation(indexes, self.algo.P, self.algo.Q, topk, pool)
        if repr:
            mo = np.int32(-1)
            topks = [[self.algo._idmanager.itemids[t] for t in tt if t != mo]
                     for tt in topks]
        return keys, topks, scores


class ParBPRMF(ParALS):
    def topk_recommendation(self, keys, topk=10, pool=None, repr=False):
        """See the documentation of Parallel."""
        if self.algo.opt._nrz_P or self.algo.opt._nrz_Q:
            raise RuntimeError('Cannot make topk recommendation with normalized factors')
        # It is possible to skip make recommendation for not-existed keys.
        indexes = self.algo.get_index_pool(keys, group='user')
        keys = [k for k, i in zip(keys, indexes) if i is not None]
        indexes = np.array([i for i in indexes if i is not None], dtype=np.int32)
        if pool is not None:
            pool = self.algo.get_index_pool(pool, group='item')
            if len(pool) == 0:
                raise RuntimeError('pool is empty')
        else:
            # It assume that empty pool menas for all items
            pool = np.array([], dtype=np.int32)
        topks, scores = super()._topk_recommendation_bias(indexes, self.algo.P, self.algo.Q, self.algo.Qb, topk, pool)
        if repr:
            topks = [[self.algo._idmanager.itemids[t] for t in tt if t != -1] for tt in topks]
        return keys, topks, scores


class ParW2V(Parallel):
    def __init__(self, algo, **kwargs):
        num_workers = int(kwargs.get('num_workers', algo.opt.num_workers))
        super().__init__(algo, num_workers=num_workers)

    def most_similar(self, keys, topk=10, pool=None, repr=False, ef_search=-1, use_mmap=True):
        """See the documentation of Parallel."""
        self.algo.normalize(group='item')
        indexes = self.algo.get_index_pool(keys, group='item')
        keys = [k for k, i in zip(keys, indexes) if i is not None]
        indexes = np.array([i for i in indexes if i is not None], dtype=np.int32)
        if pool is not None:
            pool = self.algo.get_index_pool(pool, group='item')
            if len(pool) == 0:
                raise RuntimeError('pool is empty')
        else:
            # It assume that empty pool menas for all items
            pool = np.array([], dtype=np.int32)
        topks, scores = super()._most_similar('item', indexes, self.algo.L0, topk, pool, ef_search, use_mmap)
        if repr:
            mo = np.int32(-1)
            topks = [[self.algo._idmanager.itemids[t] for t in tt if t != mo]
                     for tt in topks]
        return topks, scores

    def topk_recommendation(self, keys, topk=10, pool=None):
        raise NotImplementedError


# TODO: Re-think about CFR internal data structure.
class ParCFR(Parallel):
    pass
