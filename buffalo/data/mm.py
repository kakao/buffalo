# -*- coding: utf-8 -*-
import os

import h5py
import numpy as np

from buffalo.misc import aux, log
from buffalo.data.base import Data


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
                'batch_mb': 1024,
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
        header.create_dataset('completed', (1,), dtype='int64')
        header['num_users'][0] = num_users
        header['num_items'][0] = num_items
        header['num_nnz'][0] = num_nnz
        header['completed'][0] = 0

        iid_max_col = kwargs['iid_max_col']
        uid_max_col = kwargs['uid_max_col']
        idmap = f.create_group('idmap')
        idmap.create_dataset('rows', (num_users,), dtype='S%s' % uid_max_col,
                             maxshape=(num_users,))
        idmap.create_dataset('cols', (num_items,), dtype='S%s' % iid_max_col,
                             maxshape=(num_items,))
        return f

    def _build(self, db, mm_path, num_lines, max_key, data_chunk_size=1000000, rowwise=True):
        with open(mm_path) as fin:
            prev_key, data_index, data_rear_index = 0, 0, 0
            chunk, data_chunk = [], []
            with log.pbar(self.logger.debug, total=num_lines, mininterval=30) as pbar:
                for line in fin:
                    pbar.update(1)
                    tkns = line.strip().split()
                    if tkns[0] == '%' or len(tkns) != 3:
                        continue
                    u, i, v = int(tkns[0]) - 1, int(tkns[1]) - 1, float(tkns[2])
                    if not rowwise:
                        u, i = i, u
                    if prev_key != u:
                        for empty_index in range(prev_key + 1, u):
                            db['indptr'][empty_index] = data_index
                        sz = len(chunk)
                        data_chunk.extend(chunk)
                        data_index += sz
                        db['indptr'][prev_key] = data_index
                        chunk = []
                        prev_key += 1
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

    def _create(self, data_path, P, H):
        uid_path, iid_path, main_path = P['uid_path'], P['iid_path'], P['main_path']
        num_users, num_items, num_nnz = map(int, H.split())
        with log.pbar(self.logger.debug, total=5, mininterval=30) as pbar:
            with open(uid_path) as fin:
                uid_max_col = 0
                for l in fin:
                    uid_max_col = max(uid_max_col, len(l) + 1)
            pbar.update(1)
            with open(iid_path) as fin:
                iid_max_col = 0
                for l in fin:
                    iid_max_col = max(iid_max_col, len(l) + 1)
            pbar.update(1)
            try:
                db = self.create_database(data_path,
                                          num_users=num_users,
                                          num_items=num_items,
                                          num_nnz=num_nnz,
                                          uid_max_col=uid_max_col,
                                          iid_max_col=iid_max_col)
                idmap = db['idmap']
                with open(uid_path) as fin:
                    idmap['rows'][:] = np.loadtxt(fin, dtype='S%s' % uid_max_col)
                pbar.update(1)
                with open(iid_path) as fin:
                    idmap['cols'][:] = np.loadtxt(fin, dtype='S%s' % iid_max_col)
                pbar.update(1)
                num_header_lines = 0
                with open(main_path) as fin:
                    for line in fin:
                        if line.strip().startswith('%'):
                            num_header_lines += 1
                        else:
                            break
                pbar.update(1)
            except Exception as e:
                self.logger.error('Cannot create db: %s' % (str(e)))
                os.remove(self.path)
                raise
        return db, num_header_lines

    def create(self) -> h5py.File:
        mm_main_path = self.opt.input.main
        mm_uid_path = self.opt.input.uid
        mm_iid_path = self.opt.input.iid

        data_path = self.opt.data.path
        if os.path.isfile(data_path) and self.opt.data.use_cache:
            self.logger.info('Use cached DB on %s' % data_path)
            self.open(data_path)
            return

        self.logger.info('Create database from matrix market files')
        with open(mm_main_path) as fin:
            header = '%'
            while header.startswith('%'):
                header = fin.readline()
            self.logger.debug('Building meta part...')
            db, num_header_lines = self._create(data_path,
                                                {'main_path': mm_main_path,
                                                 'uid_path': mm_uid_path,
                                                 'iid_path': mm_iid_path},
                                                header)
            try:
                self.logger.info('Building data part...')
                num_header_lines += 1  # add metaline
                tmp_main = aux.make_temporary_file(mm_main_path, ignore_lines=num_header_lines)
                self.logger.info('Create temporary data on %s.' % tmp_main)
                aux.psort(tmp_main, key=1)
                self._build(db['rowwise'], tmp_main,
                            num_lines=db['header']['num_nnz'][0],
                            max_key=db['header']['num_users'][0], rowwise=True)
                aux.psort(tmp_main, key=2)
                self._build(db['colwise'], tmp_main,
                            num_lines=db['header']['num_nnz'][0],
                            max_key=db['header']['num_items'][0], rowwise=False)
                self.temp_files.append(tmp_main)
                db['header']['completed'][0] = 1
                db.close()
                self.handle = h5py.File(data_path, 'r')
                self.path = data_path
            except Exception as e:
                self.logger.error('Cannot create db: %s' % (str(e)))
                os.remove(self.path)
                raise
        self.logger.info('DB built on %s' % data_path)
