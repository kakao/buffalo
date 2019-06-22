# -*- coding: utf-8 -*-
import os
import abc
import warnings

import h5py
import tqdm
import numpy as np

from buffalo.misc import aux
warnings.simplefilter("ignore", ResourceWarning)


def load(opt):
    if isinstance(opt, (str,)):
        opt = aux.Option(opt)

    assert isinstance(opt, (dict, aux.Option)), 'opt must be either str, or dict/aux.Option but {}'.format(type(opt))
    if opt['type'] == 'matrix_market':
        return MatrixMarket(opt)
    raise RuntimeError('Unexpected data.type: {}'.format(opt['type']))


class MatrixMarketOptions(aux.InputOptions):
    def get_default_option(self) -> aux.Option:
        opt = {
            'type': 'matrix_market',
            'input': {
                'main': '',
                'uid': '',
                'iid': ''
            },
            'data': {
                'validation_p': 0.1,
                'use_cache': False,
                'tmp_dir': '/tmp/',
                'path': './mm.h5py'
            }
        }
        return aux.Option(opt)

    def is_valid_option(self, opt) -> bool:
        super(MatrixMarketOptions, self).is_valid_option(opt)
        if not opt['type'] == 'matrix_market':
            raise RuntimeError('Invalid data type: %s' % opt['type'])
        return True


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

    def get_header(self):
        assert self.handle, 'DB is not opened'
        if not self.header:
            self.header = {'num_nnz': self.handle['header']['num_nnz'][0],
                           'num_users': self.handle['header']['num_users'][0],
                           'num_items': self.handle['header']['num_items'][0]}
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


class MatrixMarket(Data):
    def __init__(self, opt, *args, **kwargs):
        super(MatrixMarket, self).__init__(opt, *args, **kwargs)
        self.logger = aux.get_logger('MatrixMarket')

    def create_database(self, path, **kwargs):
        f = h5py.File(path, 'w')
        self.path = path

        f.create_group('rowwise')
        f.create_group('colwise')
        num_users, num_items, num_nnz = kwargs['num_users'], kwargs['num_items'], kwargs['num_nnz']
        for g in [f['rowwise'], f['colwise']]:
            g.create_dataset('key', (num_nnz,), dtype='int32', maxshape=(num_nnz,))
            g.create_dataset('val', (num_nnz,), dtype='float32', maxshape=(num_nnz,))
        f['rowwise'].create_dataset('indptr', (num_users,), dtype='int64', maxshape=(num_users,))
        f['colwise'].create_dataset('indptr', (num_items,), dtype='int64', maxshape=(num_items,))

        header = f.create_group('header')
        header.create_dataset('num_users', (1,), dtype='int64')
        header.create_dataset('num_items', (1,), dtype='int64')
        header.create_dataset('num_nnz', (1,), dtype='int64')
        header['num_users'][0] = num_users
        header['num_items'][0] = num_items
        header['num_nnz'][0] = num_nnz

        iid_max_col = kwargs['iid_max_col']
        uid_max_col = kwargs['uid_max_col']
        idmap = f.create_group('idmap')
        idmap.create_dataset('rows', (num_users, uid_max_col), dtype='S%s' % uid_max_col,
                             maxshape=(num_users, uid_max_col))
        idmap.create_dataset('cols', (num_items, iid_max_col), dtype='S%s' % iid_max_col,
                             maxshape=(num_items, iid_max_col))
        return f

    def _build(self, db, mm_path, max_key, rowwise=True):
        with open(mm_path) as fin:
            prev_key, data_index = 0, 0
            chunk = []
            for line in tqdm.tqdm(fin, desc='build index', total=max_key, mininterval=1):
                tkns = line.strip().split()
                if tkns[0] == '%' or len(tkns) != 3:
                    continue
                u, i, v = int(tkns[0]) - 1, int(tkns[1]) - 1, float(tkns[2])
                if not rowwise:
                    u, i = i, u
                chunk.append((i, v))
                if prev_key != u:
                    for empty_index in range(prev_key + 1, u):
                        db['indptr'][empty_index] = data_index
                    sz = len(chunk)
                    db['key'][data_index:data_index + sz] = [k for k, _ in chunk]
                    db['val'][data_index:data_index + sz] = [v for _, v in chunk]
                    data_index += sz
                    db['indptr'][prev_key] = data_index
                    chunk = []
                    prev_key += 1
            if chunk:
                sz = len(chunk)
                db['key'][data_index:data_index + sz] = [k for k, _ in chunk]
                db['val'][data_index:data_index + sz] = [v for _, v in chunk]
                data_index += sz
                db['indptr'][prev_key] = data_index
                prev_key += 1
            for empty_index in range(prev_key, max_key):
                db['indptr'][empty_index] = data_index

    def create(self) -> h5py.File:
        mm_main_path = self.opt.input.main
        mm_uid_path = self.opt.input.uid
        mm_iid_path = self.opt.input.iid

        data_path = self.opt.data.path
        if os.path.isfile(data_path) and self.opt.data.use_cache:
            self.handle = h5py.File(data_path, 'r')
            self.path = data_path
            return

        self.logger.info('create database from matrix market files')
        with open(mm_main_path) as fin:
            header = '%'
            while header.startswith('%'):
                header = fin.readline()
            num_users, num_items, num_nnz = map(int, header.split())
            with open(mm_uid_path) as fin:
                uid_max_col = max([len(l.strip()) + 1 for l in fin])
            with open(mm_iid_path) as fin:
                iid_max_col = max([len(l.strip()) + 1 for l in fin])
            try:
                db = self.create_database(data_path,
                                          num_users=num_users,
                                          num_items=num_items,
                                          num_nnz=num_nnz,
                                          uid_max_col=uid_max_col,
                                          iid_max_col=iid_max_col)
                idmap = db['idmap']
                with open(mm_uid_path) as fin:
                    for idx, line in enumerate(fin):
                        idmap['rows'][idx] = np.string_(line.strip())
                with open(mm_iid_path) as fin:
                    for idx, line in enumerate(fin):
                        idmap['cols'][idx] = np.string_(line.strip())

                num_header_lines = 0
                with open(mm_main_path) as fin:
                    for line in fin:
                        if line.strip().startswith('%'):
                            num_header_lines += 1
                        else:
                            break
                num_header_lines += 1  # add metaline
                tmp_main = aux.make_temporary_file(mm_main_path, ignore_lines=num_header_lines)
                self.logger.info('create temporary data: %s' % tmp_main)
                aux.psort(tmp_main, key=1)
                self._build(db['rowwise'], tmp_main,
                            max_key=db['header']['num_users'][0], rowwise=True)
                aux.psort(tmp_main, key=2)
                self._build(db['colwise'], tmp_main,
                            max_key=db['header']['num_items'][0], rowwise=False)
                self.temp_files.append(tmp_main)
                db.close()
                self.handle = h5py.File(data_path, 'r')
                self.path = data_path
            except Exception as e:
                self.logger.error('Cannot create db: %s' % (str(e)))
                os.remove(self.path)
                raise
