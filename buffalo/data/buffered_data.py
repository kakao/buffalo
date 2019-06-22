# -*- coding: utf-8 -*-
import abc
import bisect
import numpy as np
from buffalo.misc import aux


class BufferedData(object):
    def __init__(self):
        self.logger = aux.get_logger('BufferedData')

    @abc.abstractmethod
    def initialize(self, limit):
        pass

    @abc.abstractmethod
    def reset(self):
        self.index = 0
        self.ptr_index = 0

    @abc.abstractmethod
    def get(self):
        pass
        # TODO: Checkout long long structure for Eigen/Eigency
        return (self.ptr_index,
                self.indptr,
                self.rows,
                self.keys,
                self.vals)


class BufferedDataMM(BufferedData):
    def __init__(self):
        super(BufferedDataMM, self).__init__()
        self.axis = 'rowwise'
        self.major = {'rowwise': {},
                      'colwise': {}}

    def initialize(self, limit, data):
        self.data = data
        for axis in ['rowwise', 'colwise']:
            lim = int(limit / 2)
            group = data.get_group(axis=axis)
            header = data.get_header()
            m = self.major[axis]
            m['index'] = 0
            m['limit'] = lim
            m['start_x'] = 0
            m['next_x'] = 0
            m['max_x'] = header['num_users'] if axis == 'rowwise' else header['num_items']
            m['indptr'] = group['indptr'][::]
            m['keys'] = np.zeros(shape=(lim,), dtype=np.int32, order='F')
            m['vals'] = np.zeros(shape=(lim,), dtype=np.float32, order='F')

    def fetch_batch(self):
        m = self.major[self.axis]
        while True:
            if m['start_x'] == 0 and m['next_x'] >= m['max_x']:
                self.logger.debug('already fully fetched')
                yield True
                raise StopIteration

            if m['next_x'] + 1 >= m['max_x']:
                m['start_x'], m['next_x'] = 0, 0
                raise StopIteration

            m['start_x'] = m['next_x']

            group = self.data.get_group(axis=self.axis)
            beg = 0 if m['next_x'] == 0 else m['indptr'][m['next_x'] - 1]
            where = bisect.bisect_left(m['indptr'], beg + m['limit'])
            if where == 0:
                raise RuntimeError('Need more memory to load the data')
            end = m['indptr'][where - 1]
            m['next_x'] = where - 1
            size = end - beg
            m['keys'][:size] = group['key'][beg:end]
            m['vals'][:size] = group['val'][beg:end]
            yield True

    def set_axis(self, axis):
        assert axis in ['rowwise', 'colwise'], 'Unexpected axis: {}'.format(axis)
        self.axis = axis

    def reset(self):
        for m in self.major.valus():
            m['index'], m['start_x'], m['next_x'] = 0, 0

    def get(self):
        m = self.major[self.axis]
        return (m['start_x'],
                m['next_x'],
                m['indptr'],
                m['keys'],
                m['vals'])
