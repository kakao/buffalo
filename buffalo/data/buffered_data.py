# -*- coding: utf-8 -*-
import abc
import bisect
import numpy as np
from buffalo.misc import log


class BufferedData(object):
    def __init__(self):
        self.logger = log.get_logger('BufferedData')

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

    def initialize(self, data):
        self.data = data
        # 16 bytes(indptr8, keys4, vals4)
        limit = max(int(((self.data.opt.data.batch_mb * 1000 * 1000) / 16.)), 64)
        minimum_required_batch_size = 0
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
            minimum_required_batch_size = max([m['indptr'][i] - m['indptr'][i - 1]
                                               for i in range(1, len(m['indptr']))])
            m['keys'] = np.zeros(shape=(lim,), dtype=np.int32, order='F')
            m['vals'] = np.zeros(shape=(lim,), dtype=np.float32, order='F')
        if minimum_required_batch_size > int(limit / 2):
            self.logger.warning('Increased batch size due to '
                                'minimum required batch size is %d for the data, but %d given. '
                                'Increasing batch_mb would be helpful for faster traininig.',
                                minimum_required_batch_size, int(limit / 2))
            for axis in ['rowwise', 'colwise']:
                m = self.major[axis]
                lim = minimum_required_batch_size + 1
                m['limit'] = lim
                m['keys'] = np.zeros(shape=(lim,), dtype=np.int32, order='F')
                m['vals'] = np.zeros(shape=(lim,), dtype=np.float32, order='F')

    def fetch_batch(self):
        m = self.major[self.axis]
        while True:
            if m['start_x'] == 0 and m['next_x'] + 1 >= m['max_x']:
                yield m['indptr'][-1]
                raise StopIteration

            if m['next_x'] + 1 >= m['max_x']:
                m['start_x'], m['next_x'] = 0, 0
                raise StopIteration

            m['start_x'] = m['next_x']

            group = self.data.get_group(axis=self.axis)
            beg = 0 if m['start_x'] == 0 else m['indptr'][m['start_x'] - 1]
            where = bisect.bisect_left(m['indptr'], beg + m['limit'])
            if where == m['start_x']:
                current_batch_size = m['limit']
                need_batch_size = m['indptr'][where] - beg
                raise RuntimeError('Need more memory to load the data, '
                                   'current buffer size is %d but need to be at least %d. '
                                   'Increase batch_mb value.' % (current_batch_size, need_batch_size))
            end = m['indptr'][where - 1]
            m['next_x'] = where
            size = end - beg
            m['keys'][:size] = group['key'][beg:end]
            m['vals'][:size] = group['val'][beg:end]
            yield size

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
