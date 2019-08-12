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


class BufferedDataMatrix(BufferedData):
    """Buffered Data for MatrixMarket

    This class feed chunked data to training step.
    """
    def __init__(self):
        super().__init__()
        self.group = 'rowwise'
        self.major = {'rowwise': {}, 'colwise': {}, 'sppmi': {}}

    def free(self, buf):
        buf['indptr'] = None
        buf['keys'] = None
        buf['vals'] = None

    def initialize(self, data, with_sppmi=False):
        self.data = data
        # 16 bytes(indptr8, keys4, vals4)
        limit = max(int(((self.data.opt.data.batch_mb * 1024 * 1024) / 16.)), 64)
        minimum_required_batch_size = 0
        Gs = ['rowwise', 'colwise']
        if with_sppmi:
            Gs.append('sppmi')
        for G in Gs:
            lim = int(limit / 2)
            group = data.get_group(G)
            header = data.get_header()
            if self.major[G]:
                self.free(self.major[G])
            m = self.major[G]
            m['index'] = 0
            m['limit'] = lim
            m['start_x'] = 0
            m['next_x'] = 0
            m['max_x'] = header['num_users'] if G == 'rowwise' else header['num_items']
            m['indptr'] = group['indptr'][::]
            minimum_required_batch_size = max([m['indptr'][i] - m['indptr'][i - 1]
                                               for i in range(1, len(m['indptr']))])
            m['keys'] = np.zeros(shape=(lim,), dtype=np.int32, order='C')
            m['vals'] = np.zeros(shape=(lim,), dtype=np.float32, order='C')
        self.logger.info(f'Set data buffer size as {limit}(minimum required batch size is {minimum_required_batch_size}).')
        if minimum_required_batch_size > int(limit / 2):
            self.logger.warning('Given batch size(%d) is smaller than '
                                'minimum required batch size(%d) for the data. '
                                'Increasing batch_mb would be helpful for faster traininig.',
                                int(limit / 2), minimum_required_batch_size)
            for G in ['rowwise', 'colwise']:
                m = self.major[G]
                lim = minimum_required_batch_size + 1
                m['limit'] = lim
                m['keys'] = np.zeros(shape=(lim,), dtype=np.int32, order='C')
                m['vals'] = np.zeros(shape=(lim,), dtype=np.float32, order='C')

    def fetch_batch(self):
        m = self.major[self.group]
        flushed = False
        while True:
            if m['start_x'] == 0 and m['next_x'] + 1 >= m['max_x']:
                if not flushed:
                    yield m['indptr'][-1]
                raise StopIteration

            if m['next_x'] + 1 >= m['max_x']:
                m['start_x'], m['next_x'] = 0, 0
                raise StopIteration

            m['start_x'] = m['next_x']

            group = self.data.get_group(self.group)
            beg = 0 if m['start_x'] == 0 else m['indptr'][m['start_x'] - 1]
            where = bisect.bisect_left(m['indptr'], beg + m['limit'])
            if where == m['start_x']:
                current_batch_size = m['limit']
                need_batch_size = m['indptr'][where] - beg
                raise RuntimeError('Need more memory to load the data, '
                                   'cannot load data with buffer size %d that should be at least %d. '
                                   'Increase batch_mb value to deal with this.' % (current_batch_size, need_batch_size))
            end = m['indptr'][where - 1]
            m['next_x'] = where
            size = end - beg
            m['keys'][:size] = group['key'][beg:end]
            m['vals'][:size] = group['val'][beg:end]
            if m['next_x'] + 1 >= m['max_x']:
                flushed = True
            yield size

    def set_group(self, group):
        assert group in ['rowwise', 'colwise', 'sppmi'], 'Unexpected group: {}'.format(group)
        self.group = group

    def reset(self):
        for m in self.major.valus():
            m['index'], m['start_x'], m['next_x'] = 0, 0, 0

    def get(self):
        m = self.major[self.group]
        return (m['start_x'],
                m['next_x'],
                m['indptr'],
                m['keys'],
                m['vals'])


class BufferedDataStream(BufferedData):
    """Buffered Data for Stream

    This class feed chunked data to training step.
    """
    def __init__(self):
        super().__init__()
        self.major = {'rowwise': {}}
        self.group = 'rowwise'

    def free(self, buf):
        buf['indptr'] = None
        buf['keys'] = None

    def initialize(self, data):
        self.data = data
        assert self.data.data_type == 'stream'
        # 12 bytes(indptr8, keys4)
        limit = max(int(((self.data.opt.data.batch_mb * 1000 * 1000) / 12.)), 64)
        minimum_required_batch_size = 0
        lim = int(limit / 2)
        G = 'rowwise'
        group = data.get_group(G)
        header = data.get_header()
        m = self.major[G]
        if self.major[G]:
            self.free(self.major[G])
        m['index'] = 0
        m['limit'] = lim
        m['start_x'] = 0
        m['next_x'] = 0
        m['max_x'] = header['num_users']
        m['indptr'] = group['indptr'][::]
        minimum_required_batch_size = max([m['indptr'][i] - m['indptr'][i - 1]
                                           for i in range(1, len(m['indptr']))])
        m['keys'] = np.zeros(shape=(lim,), dtype=np.int32, order='F')
        if minimum_required_batch_size > int(limit / 2):
            self.logger.warning('Given batch size(%d) is smaller than '
                                'minimum required batch size(%d) is for the data. '
                                'Increasing batch_mb would be helpful for faster traininig.',
                                int(limit / 2), minimum_required_batch_size)
            m = self.major[G]
            lim = minimum_required_batch_size + 1
            m['limit'] = lim
            m['keys'] = np.zeros(shape=(lim,), dtype=np.int32, order=order)

    def fetch_batch(self):
        m = self.major[self.group]
        flushed = False
        while True:
            if m['start_x'] == 0 and m['next_x'] + 1 >= m['max_x']:
                if not flushed:
                    yield m['indptr'][-1]
                raise StopIteration

            if m['next_x'] + 1 >= m['max_x']:
                m['start_x'], m['next_x'] = 0, 0
                raise StopIteration

            m['start_x'] = m['next_x']

            group = self.data.get_group(self.group)
            beg = 0 if m['start_x'] == 0 else m['indptr'][m['start_x'] - 1]
            where = bisect.bisect_left(m['indptr'], beg + m['limit'])
            if where == m['start_x']:
                current_batch_size = m['limit']
                need_batch_size = m['indptr'][where] - beg
                raise RuntimeError('Need more memory to load the data, '
                                   'cannot load data with buffer size %d that should be at least %d. '
                                   'Increase batch_mb value to deal with this.' % (current_batch_size, need_batch_size))
            end = m['indptr'][where - 1]
            m['next_x'] = where
            size = end - beg
            m['keys'][:size] = group['key'][beg:end]
            if m['next_x'] + 1 >= m['max_x']:
                flushed = True
            yield size

    def set_group(self, group):
        assert group in ['rowwise', 'sppmi'], 'Unexpected group: {}'.format(group)
        self.group = group

    def reset(self):
        for m in self.major.valus():
            m['index'], m['start_x'], m['next_x'] = 0, 0, 0

    def get(self):
        m = self.major[self.group]
        return (m['start_x'],
                m['next_x'],
                m['indptr'],
                m['keys'])
