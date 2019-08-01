# -*- coding: utf-8 -*-
import os
import abc

import h5py
import numpy as np

from buffalo.data import prepro
from buffalo.misc import aux, log


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
        self.data_type = None
        self.id_mapped = False
        self.userid_map, self.itemid_map = {}, {}

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
            self.header = {'num_nnz': self.handle.attrs['num_nnz'],
                           'num_users': self.handle.attrs['num_users'],
                           'num_items': self.handle.attrs['num_items'],
                           'completed': self.handle.attrs['completed']}
        return self.header

    def build_itemid_map(self):
        idmap = self.get_group('idmap')
        header = self.get_header()
        if idmap['cols'].shape[0] == 0:
            self.itemids = list(map(str, list(range(header['num_items']))))
            self.itemid_map = {str(i): i for i in range(header['num_items'])}
        else:
            self.itemids = list(map(lambda x: x.decode('utf-8', 'ignore'), idmap['cols'][::]))
            self.itemid_map = {v: idx
                               for idx, v in enumerate(self.itemids)}

    def build_userid_map(self):
        idmap = self.get_group('idmap')
        header = self.get_header()
        if idmap['rows'].shape[0] == 0:
            self.userids = list(map(str, list(range(header['num_users']))))
            self.userid_map = {str(i): i for i in range(header['num_users'])}
        else:
            self.userids = list(map(lambda x: x.decode('utf-8', 'ignore'), idmap['rows'][::]))
            self.userid_map = {v: idx
                               for idx, v in enumerate(self.userids)}

    def build_idmaps(self):
        self.id_mapped = True
        self.build_itemid_map()
        self.build_userid_map()

    def get_group(self, group_name='rowwise'):
        assert group_name in ['rowwise', 'colwise', 'vali', 'idmap'], 'Unexpected group_name: {}'.format(group_name)
        assert self.handle, 'DB is not opened'
        group = self.handle[group_name]
        return group

    def has_group(self, name):
        return name in self.handle

    def iterate(self, axis='rowwise', use_repr_name=False) -> [int, int, float]:
        """Iterate over datas

        args:
            group: which data group
            use_repr_name: return representive name of internal ids
        """
        if use_repr_name and not self.id_mapped:
            raise RuntimeError('IDMaps not built')
        userids, itemids = None, None
        if use_repr_name:
            userids, itemids = self.userids, self.itemids
            if axis == 'colwise':
                userids, itemids = itemids, userids

        if self.opt.data.internal_data_type == 'matrix':
            assert axis in ['rowwise', 'colwise'], 'Unexpected data axis: {}'.format(axis)
            assert self.handle, 'DB is not opened'
            group = self.handle[axis]
            data_index = 0
            for u, end in enumerate(group['indptr']):
                keys = group['key'][data_index:end]
                vals = group['val'][data_index:end]
                if use_repr_name:
                    for k, v in zip(keys, vals):
                        yield userids[u], itemids[k], v
                else:
                    for k, v in zip(keys, vals):
                        yield u, k, v
                data_index = end
        elif self.opt.data.internal_data_type == 'stream':
            assert axis in ['rowwise'], 'Unexpected date axis: {}'.format(axis)
            assert self.handle, 'DB is not opened'
            group = self.handle[axis]
            data_index = 0
            for u, end in enumerate(group['indptr']):
                keys = group['key'][data_index:end]
                if use_repr_name:
                    for k in keys:
                        yield userids[u], itemids[k]
                else:
                    for k in keys:
                        yield u, k
                data_index = end

    def get(self, index, axis='rowwise') -> [int, int, float]:
        if self.opt.data.internal_data_type == 'matrix':
            assert axis in ['rowwise', 'colwise'], 'Unexpected data axis: {}'.format(axis)
            assert self.handle, 'DB is not opened'
            group = self.handle[axis]
            begin = 0 if index == 0 else group['indptr'][index - 1]
            end = group['indptr'][index]
            keys = group['key'][begin:end]
            vals = group['val'][begin:end]
            return (keys, vals)
        elif self.opt.data.internal_data_type == 'stream':
            assert axis in ['rowwise'], 'Unexpected date axis: {}'.format(axis)
            assert self.handle, 'DB is not opened'
            group = self.handle[axis]
            begin = 0 if index == 0 else group['indptr'][index - 1]
            end = group['indptr'][index]
            keys = group['key'][begin:end]
            return (keys,)

    def __del__(self):
        if self.handle:
            self.handle = None
            self.header = None
        for path in self.temp_files:
            os.remove(path)

    def close(self):
        if self.handle:
            self.handle = None
            self.header = None

    def _create_database(self, path, **kwargs):
        # Create database structure
        f = h5py.File(path, 'w')
        self.path = path
        num_users, num_items, num_nnz = kwargs['num_users'], kwargs['num_items'], kwargs['num_nnz']

        f.create_group('rowwise')
        f.create_group('colwise')
        for g in [f['rowwise'], f['colwise']]:
            g.create_dataset('key', (num_nnz,), dtype='int32', maxshape=(num_nnz,))
            g.create_dataset('val', (num_nnz,), dtype='float32', maxshape=(num_nnz,))
        f['rowwise'].create_dataset('indptr', (num_users,), dtype='int64', maxshape=(num_users,))
        f['colwise'].create_dataset('indptr', (num_items,), dtype='int64', maxshape=(num_items,))

        num_nnz = self._create_validation(f, **kwargs)

        f.attrs['num_users'] = num_users
        f.attrs['num_items'] = num_items
        f.attrs['num_nnz'] = num_nnz
        f.attrs['completed'] = 0

        iid_max_col = kwargs['iid_max_col']
        uid_max_col = kwargs['uid_max_col']
        idmap = f.create_group('idmap')
        idmap.create_dataset('rows', (num_users,), dtype='S%s' % uid_max_col,
                             maxshape=(num_users,))
        idmap.create_dataset('cols', (num_items,), dtype='S%s' % iid_max_col,
                             maxshape=(num_items,))
        return f

    def _create_validation(self, f, **kwargs):
        if not self.opt.data.validation:
            return kwargs['num_nnz']
        num_nnz = kwargs['num_nnz']
        method = self.opt.data.validation.name

        f.create_group('vali')
        g = f['vali']
        g.attrs['method'] = method
        g.attrs['n'] = 0
        if method == 'sample':
            sz = min(self.opt.data.validation.max_samples,
                    int(num_nnz * self.opt.data.validation.p))
            g.create_dataset('row', (sz,), dtype='int32', maxshape=(sz,))
            g.create_dataset('col', (sz,), dtype='int32', maxshape=(sz,))
            g.create_dataset('val', (sz,), dtype='float32', maxshape=(sz,))
            g.create_dataset('indexes', (sz,), dtype='int64', maxshape=(sz,))
            g['indexes'][:] = np.random.choice(num_nnz, sz, replace=False)
            num_nnz -= sz
        elif method in ['newest']:
            sz = kwargs['num_validation_samples']
            g.create_dataset('row', (sz,), dtype='int32', maxshape=(sz,))
            g.create_dataset('col', (sz,), dtype='int32', maxshape=(sz,))
            g.create_dataset('val', (sz,), dtype='float32', maxshape=(sz,))
            g.attrs['n'] = self.opt.data.validation.n
            # We don't need to reduce sample size for validation samples. It
            # already applied on caller side.
        g.attrs['num_samples'] = sz
        return num_nnz

    def fill_validation_data(self, db, validation_data):
        if not validation_data:
            return
        validation_data = [line.strip().split() for line in validation_data]
        assert len(validation_data) == db['vali'].attrs['num_samples'], 'Given data data is not matched with required for validation data'
        db['vali']['row'][:] = [int(r) - 1 for r, _, _ in validation_data]  # 0-based
        db['vali']['col'][:] = [int(c) - 1 for _, c, _ in validation_data]  # 0-based
        db['vali']['val'][:] = [self.value_prepro(float(v)) for _, _, v in validation_data]

    def _build_compressed_triplets(self, db, mm_path, num_lines, max_key, data_chunk_size=1000000, switch_row_col=False):
        # NOTE: this part is the bottle-neck, can we do better?
        with open(mm_path) as fin:
            prev_key, data_index, data_rear_index = 0, 0, 0
            chunk, data_chunk = [], []
            for line in log.iter_pbar(log.DEBUG, iterable=fin, total=num_lines, mininterval=30):
                tkns = line.strip().split()
                if tkns[0] == '%' or len(tkns) != 3:
                    continue
                u, i, v = int(tkns[0]) - 1, int(tkns[1]) - 1, float(tkns[2])
                v = self.value_prepro(v)
                if switch_row_col:
                    u, i = i, u
                if prev_key != u:
                    sz = len(chunk)
                    data_chunk.extend(chunk)
                    data_index += sz
                    db['indptr'][prev_key] = data_index
                    chunk = []
                    for empty_index in range(prev_key + 1, u):
                        db['indptr'][empty_index] = data_index
                    prev_key = u

                chunk.append((i, v))
                if len(data_chunk) % data_chunk_size == 0:
                    sz = len(data_chunk)
                    db['key'][data_rear_index:data_rear_index + sz] = [k for k, _ in data_chunk]
                    db['val'][data_rear_index:data_rear_index + sz] = [v for _, v in data_chunk]
                    data_rear_index += sz
                    data_chunk = []
            if chunk:
                sz = len(chunk)
                data_chunk.extend(chunk)
                data_index += sz
                db['indptr'][prev_key] = data_index
                prev_key += 1
            if len(data_chunk) > 0:
                sz = len(data_chunk)
                db['key'][data_rear_index:data_rear_index + sz] = [k for k, _ in data_chunk]
                db['val'][data_rear_index:data_rear_index + sz] = [v for _, v in data_chunk]
                data_rear_index += sz
                data_chunk = []
            for empty_index in range(prev_key, max_key):
                db['indptr'][empty_index] = data_index


class DataOption(object):
    def is_valid_option(self, opt) -> bool:
        """General type/logic checking"""

        if 'validation' in opt['data']:
            assert opt['data']['validation']['name'] in ['sample', 'newest'], 'Unknown validation.name.'
            if opt['data']['validation']['name'] == 'sample':
                assert hasattr(opt['data']['validation'], 'max_samples'), 'max_samples not defined on data.validation.'
                assert isinstance(opt['data']['validation']['max_samples'], int), 'invalid type for data.validation.max_samples'
                assert hasattr(opt['data']['validation'], 'p'), 'not defined on data.validation.p'
                assert isinstance(opt['data']['validation']['p'], float), 'invalid type for data.validation.p'
            if opt['data']['validation']['name'] in ['newest']:
                assert hasattr(opt['data']['validation'], 'max_samples'), 'max_samples not defined on data.validation.'
                assert isinstance(opt['data']['validation']['max_samples'], int), 'invalid type for data.validation.max_samples'
                assert hasattr(opt['data']['validation'], 'n'), 'not defined on data.validation.n'
                assert isinstance(opt['data']['validation']['n'], int), 'invalid type for data.validation.n'
        return True
