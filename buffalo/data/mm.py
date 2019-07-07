# -*- coding: utf-8 -*-
import os

import h5py
import numpy as np

from buffalo.data import prepro
from buffalo.misc import aux, log
from buffalo.data.base import Data, DataOption


class MatrixMarketOptions(DataOption):
    def get_default_option(self) -> aux.Option:
        opt = {
            'type': 'matrix_market',
            'input': {
                'main': '',
                'uid': '',  # if not set, row-id is used as userid.
                'iid': ''  # if not set, col-id is used as itemid.
            },
            'data': {
                'validation': {
                    'name': 'sample',
                    'p': 0.01,
                    'max_samples': 500
                },
                'batch_mb': 1024,
                'validation_p': 0.1,
                'use_cache': False,
                'tmp_dir': '/tmp/',
                'path': './mm.h5py'
            }
        }
        return aux.Option(opt)

    def is_valid_option(self, opt) -> bool:
        assert super(MatrixMarketOptions, self).is_valid_option(opt)
        if not opt['type'] == 'matrix_market':
            raise RuntimeError('Invalid data type: %s' % opt['type'])
        return True


class MatrixMarket(Data):
    def __init__(self, opt, *args, **kwargs):
        super(MatrixMarket, self).__init__(opt, *args, **kwargs)
        self.name = 'MatrixMarket'
        self.logger = log.get_logger('MatrixMarket')
        if isinstance(self.value_prepro,
                      (prepro.SPPMI)):
            raise RuntimeError(f'{self.opt.data.value_prepro.name} does not support MatrixMarket')

    def create_database(self, path, **kwargs):
        # Create database structure
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

        if self.opt.data.validation:
            f.create_group('vali')
            g = f['vali']
            sz = min(self.opt.data.validation.max_samples, int(num_nnz * self.opt.data.validation.p))
            g.create_dataset('row', (sz,), dtype='int32', maxshape=(sz,))
            g.create_dataset('col', (sz,), dtype='int32', maxshape=(sz,))
            g.create_dataset('val', (sz,), dtype='float32', maxshape=(sz,))
            g.create_dataset('indexes', (sz,), dtype='int64', maxshape=(sz,))
            g['indexes'][:] = np.random.choice(num_nnz, sz, replace=False)
            num_nnz -= sz

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
        # NOTE: this part is the bottle-neck, can we do better?
        with open(mm_path) as fin:
            prev_key, data_index, data_rear_index = 0, 0, 0
            chunk, data_chunk = [], []
            with log.pbar(log.DEBUG, total=num_lines, mininterval=30) as pbar:
                for line in fin:
                    pbar.update(1)
                    tkns = line.strip().split()
                    if tkns[0] == '%' or len(tkns) != 3:
                        continue
                    u, i, v = int(tkns[0]) - 1, int(tkns[1]) - 1, float(tkns[2])
                    v = self.value_prepro(v)
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
        def get_max_column_length(fname):
            with open(fname) as fin:
                max_col = 0
                for l in fin:
                    max_col = max(max_col, len(l))
            return max_col

        uid_path, iid_path, main_path = P['uid_path'], P['iid_path'], P['main_path']
        num_users, num_items, num_nnz = map(int, H.split())
        # Manually updating progress bar is a bit naive
        with log.pbar(log.DEBUG, total=5, mininterval=30) as pbar:
            uid_max_col = len(str(num_users)) + 1
            if uid_path:
                uid_max_col = get_max_column_length(uid_path) + 1
            pbar.update(1)
            iid_max_col = len(str(num_items)) + 1
            if iid_path:
                iid_max_col = get_max_column_length(iid_path) + 1
            pbar.update(1)
            try:
                db = self.create_database(data_path,
                                          num_users=num_users,
                                          num_items=num_items,
                                          num_nnz=num_nnz,
                                          uid_max_col=uid_max_col,
                                          iid_max_col=iid_max_col)
                idmap = db['idmap']
                # if not given, assume id as is
                if uid_path:
                    with open(uid_path) as fin:
                        idmap['rows'][:] = np.loadtxt(fin, dtype=f'S{uid_max_col}')
                else:
                    idmap['rows'][:] = np.array([str(i) for i in range(1, num_users + 1)],
                                                dtype=f'S{uid_max_col}')
                pbar.update(1)
                if iid_path:
                    with open(iid_path) as fin:
                        idmap['cols'][:] = np.loadtxt(fin, dtype=f'S{iid_max_col}')
                else:
                    idmap['cols'][:] = np.array([str(i) for i in range(1, num_items + 1)],
                                                dtype=f'S{iid_max_col}')
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

    def fill_validation_data(self, db, validation_data):
        if not validation_data:
            return
        validation_data = [line.strip().split() for line in validation_data]
        assert len(validation_data) == db['vali']['indexes'].shape[0], 'Given data data is not matched with required for validation data'
        db['vali']['row'][:] = [int(r) - 1 for r, _, _ in validation_data]  # 0-based
        db['vali']['col'][:] = [int(c) - 1 for _, c, _ in validation_data]  # 0-based
        db['vali']['val'][:] = [self.value_prepro(float(v)) for _, _, v in validation_data]

    def _build_data(self, db, working_data_path, validation_data):
        aux.psort(working_data_path, key=1)
        self.prepro.pre(db['header'])
        self._build(db['rowwise'], working_data_path,
                    num_lines=db['header']['num_nnz'][0],
                    max_key=db['header']['num_users'][0], rowwise=True)
        self.prepro.post(db['rowwise'])

        self.fill_validation_data(db, validation_data)

        aux.psort(working_data_path, key=2)
        self.prepro.pre(db['header'])
        self._build(db['colwise'], working_data_path,
                    num_lines=db['header']['num_nnz'][0],
                    max_key=db['header']['num_items'][0], rowwise=False)
        self.prepro.post(db['colwise'])

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
                num_header_lines += 1  # add metaline
                self.logger.info('Creating working data...')
                pickup_line_indexes = [] if 'vali' not in db else db['vali']['indexes']
                tmp_main, validation_data = aux.make_temporary_file(mm_main_path,
                                                                    ignore_lines=num_header_lines,
                                                                    pickup_line_indexes=pickup_line_indexes)
                self.logger.debug(f'Working data is created on {tmp_main}')
                self.logger.info('Building data part...')
                self._build_data(db, tmp_main, validation_data)
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
