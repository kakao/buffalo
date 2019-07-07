# -*- coding: utf-8 -*-
import os
import abc

import h5py

from buffalo.misc import aux
from buffalo.data import prepro


class Data(object):
    def __init__(self, opt, *args, **kwargs):
        self.opt = aux.Option(opt)
        self.tmp_root = opt.data.tmp_dir
        if not os.path.isdir(self.tmp_root):
            aux.mkdirs(self.tmp_root)
        self.handle = None
        self.header = None
        self.temp_files = []
        self.prepro = prepro.PreProcess(self.opt.data)
        self.value_prepro = self.prepro
        if self.opt.data.value_prepro:
            self.prepro = getattr(prepro, self.opt.data.value_prepro.name)(self.opt.data.value_prepro)
            self.value_prepro = self.prepro

    @abc.abstractmethod
    def create_database(self, filename, **kwargs):
        pass

    def show_info(self):
        header = self.get_header()
        g = self.get_group('vali')
        info = '{name} Header({users}, {items}, {nnz}) Validation({vali} samples)'
        info = info.format(name=self.name,
                           users=header['num_users'],
                           items=header['num_items'],
                           nnz=header['num_nnz'],
                           vali=g['indexes'].shape[0])
        return info

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

    def get_group(self, group_name='rowwise'):
        assert group_name in ['rowwise', 'colwise', 'vali'], 'Unexpected group_name: {}'.format(group_name)
        assert self.handle, 'DB is not opened'
        group = self.handle[group_name]
        return group

    def has_group(self, name):
        return name in self.handle

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


class DataOption(object):
    def is_valid_option(self, opt) -> bool:
        assert super(DataOption, self).is_valid_option(opt)
        if 'validation' in opt['data']:
            assert opt['data']['validation']['name'] in ['sample'], 'Unknown validation.name.'
            if opt['data']['validation']['name'] == 'sample':
                assert hasattr(opt['data']['validation'], 'max_samples'), 'max_samples not defined on data.validation.'
                assert isinstance(opt['data']['validation']['max_samples'], int), 'invalid type for data.validation.max_samples'
                assert hasattr(opt['data']['validation'], 'p'), 'not defined on data.validation.'
                assert isinstance(opt['data']['validation']['p'], float), 'invalid type for data.validation.p'
        return True
