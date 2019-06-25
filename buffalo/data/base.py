# -*- coding: utf-8 -*-
import os
import abc

import h5py

from buffalo.misc import aux


class Data(object):
    def __init__(self, opt, *args, **kwargs):
        self.opt = aux.Option(opt)
        self.tmp_root = opt.data.tmp_dir
        if not os.path.isdir(self.tmp_root):
            aux.mkdirs(self.tmp_root)
        self.handle = None
        self.header = None
        self.temp_files = []

    @abc.abstractmethod
    def create_database(self, filename, **kwargs):
        pass

    def open(self, data_path):
        self.handle = h5py.File(data_path, 'r')
        self.path = data_path
        self.verify()

    def verify(self):
        assert self.handle, 'DB is not opened'
        if self.get_header()['completed'] != 1:
            raise RuntimeError('DB is corrupted or partially built. Please try again, after remove it.')

    def get_header(self):
        assert self.handle, 'DB is not opened'
        if not self.header:
            self.header = {'num_nnz': self.handle['header']['num_nnz'][0],
                           'num_users': self.handle['header']['num_users'][0],
                           'num_items': self.handle['header']['num_items'][0],
                           'completed': self.handle['header']['completed'][0]}
        return self.header

    def get_group(self, axis='rowwise'):
        assert axis in ['rowwise', 'colwise'], 'Unexpected axis: {}'.format(axis)
        assert self.handle, 'DB is not opened'
        group = self.handle[axis]
        return group

    def iterate(self, axis='rowwise') -> [int, int, float]:
        assert axis in ['rowwise', 'colwise'], 'Unexpected axis: {}'.format(axis)
        assert self.handle, 'DB is not opened'
        group = self.handle[axis]
        data_index = 0
        for u, end in enumerate(group['indptr']):
            keys = group['key'][data_index:end]
            vals = group['val'][data_index:end]
            for k, v in zip(keys, vals):
                yield u, k, v
            data_index = end

    def __del__(self):
        if self.handle:
            self.handle.close()
            self.handle = None
            self.header = None
        for path in self.temp_files:
            os.remove(path)

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None
            self.header = None
