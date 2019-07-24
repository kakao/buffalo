# -*- coding: utf-8 -*-
import os
import warnings
import tempfile
import traceback

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
                'internal_data_type': 'matrix',
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
        if opt['data']['internal_data_type'] != 'matrix':
            raise RuntimeError('MatrixMarket only support internal data type(matrix)')
        return True


class MatrixMarket(Data):
    def __init__(self, opt, *args, **kwargs):
        super().__init__(opt, *args, **kwargs)
        self.name = 'MatrixMarket'
        self.logger = log.get_logger('MatrixMarket')
        if isinstance(self.value_prepro,
                      (prepro.SPPMI)):
            raise RuntimeError(f'{self.opt.data.value_prepro.name} does not support MatrixMarket')
        self.data_type = 'matrix'

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
                db = self._create_database(data_path,
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
                self.logger.error(traceback.format_exc())
                raise
        return db, num_header_lines

    def _build_data(self, db, working_data_path, validation_data):
        aux.psort(working_data_path, key=1)
        self.prepro.pre(db)
        self._build_compressed_triplets(db['rowwise'], working_data_path,
                                        num_lines=db.attrs['num_nnz'],
                                        max_key=db.attrs['num_users'])
        self.prepro.post(db['rowwise'])

        self.fill_validation_data(db, validation_data)

        aux.psort(working_data_path, key=2)
        self.prepro.pre(db)
        self._build_compressed_triplets(db['colwise'], working_data_path,
                                        num_lines=db.attrs['num_nnz'],
                                        max_key=db.attrs['num_items'],
                                        switch_row_col=True)
        self.prepro.post(db['colwise'])

    def _create_working_data(self, db, source_path, ignore_lines):
        """
        Args:
            source_path: source data file path
            ignore_lines: number of lines to skip from start line
        """
        vali_indexes = [] if 'vali' not in db else db['vali']['indexes']
        vali_lines = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as w:
                fin = open(source_path, mode='r')
                for _ in range(ignore_lines):
                    fin.readline()
                target_indexes = set(vali_indexes)
                for idx, line in enumerate(fin):
                    if idx in target_indexes:
                        vali_lines.append(line.strip())
                    else:
                        w.write(line)
                w.close()
                return w.name, vali_lines

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
                tmp_main, validation_data = self._create_working_data(db,
                                                                      mm_main_path,
                                                                      num_header_lines)
                self.logger.debug(f'Working data is created on {tmp_main}')
                self.logger.info('Building data part...')
                self._build_data(db, tmp_main, validation_data)
                self.temp_files.append(tmp_main)
                db.attrs['completed'] = 1
                db.close()
                self.handle = h5py.File(data_path, 'r')
                self.path = data_path
            except Exception as e:
                self.logger.error('Cannot create db: %s' % (str(e)))
                self.logger.error(traceback.format_exc().splitlines())
                raise
            finally:
                if hasattr(self, 'patr'):
                    if os.path.isfile(self.path):
                        os.remove(self.path)
        self.logger.info('DB built on %s' % data_path)
