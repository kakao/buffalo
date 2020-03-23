# -*- coding: utf-8 -*-
import os
import abc
import psutil

import h5py
import numpy as np
from scipy.sparse import csr_matrix
from buffalo.data import prepro
from buffalo.misc import aux, log
from buffalo.data.fileio import chunking_into_bins, sort_and_compressed_binarization


class Data(object):
    def __init__(self, opt, *args, **kwargs):
        self.opt = aux.Option(opt)
        self.tmp_root = opt.data.tmp_dir
        if not os.path.isdir(self.tmp_root):
            os.makedirs(self.tmp_root)
        self.handle = None
        self.header = None
        self.prepro = prepro.PreProcess(self.opt.data)
        self.value_prepro = self.prepro
        if self.opt.data.value_prepro:
            self.prepro = getattr(prepro, self.opt.data.value_prepro.name)(self.opt.data.value_prepro)
            self.value_prepro = self.prepro
        self.data_type = None

    @abc.abstractmethod
    def create_database(self, filename, **kwargs):
        pass

    def show_info(self):
        header = self.get_header()
        vali_size = 0
        if self.has_group('vali'):
            g = self.get_group('vali')
            vali_size = g.attrs['num_samples']
        info = '{name} Header({users}, {items}, {nnz}) Validation({vali} samples)'
        info = info.format(name=self.name,
                           users=header['num_users'],
                           items=header['num_items'],
                           nnz=header['num_nnz'],
                           vali=vali_size)
        return info

    def open(self, data_path):
        self.handle = h5py.File(data_path, 'r')
        self.path = data_path
        self.verify()

    def verify(self):
        assert self.handle, 'Database is not opened'
        if self.get_header()['completed'] != 1:
            raise RuntimeError('Database is corrupted or partially built. Please try again, after remove it.')

    def get_header(self):
        assert self.handle, 'Database is not opened'
        if not self.header:
            self.header = {'num_nnz': self.handle.attrs['num_nnz'],
                           'num_users': self.handle.attrs['num_users'],
                           'num_items': self.handle.attrs['num_items'],
                           'completed': self.handle.attrs['completed']}
        return self.header

    def get_scale_info(self, with_sppmi=False, chunk_size=100000):
        ret = {k: self.handle.attrs[k] for k in ["num_users", "num_items", "num_nnz", "sppmi_nnz"]}
        if with_sppmi:
            ret["sppmi_nnz"] = self.handle.attrs["sppmi_nnz"]
        db = self.handle["rowwise"]
        num_nnz = ret["num_nnz"]
        vsum = 0.0
        for offset in range(0, num_nnz, chunk_size):
            limit = min(num_nnz, offset + chunk_size)
            vsum += np.sum(db["val"][offset: limit])
        ret["vsum"] = vsum
        return ret

    def get_group(self, group_name='rowwise'):
        assert group_name in ['rowwise', 'colwise', 'vali', 'idmap', 'sppmi'], 'Unexpected group_name: {}'.format(group_name)
        assert self.handle, 'DB is not opened'
        group = self.handle[group_name]
        return group

    def has_group(self, name):
        return name in self.handle

    def _iterate_matrix(self, axis, use_repr_name, userids, itemids):
        assert axis in ['rowwise', 'colwise'], 'Unexpected data axis: {}'.format(axis)
        assert self.handle, 'Database is not opened'
        group = self.handle[axis]
        data_index = 0
        for u, end in enumerate(group['indptr']):
            keys = group['key'][data_index:end]
            vals = group['val'][data_index:end]
            if use_repr_name:
                for k, v in zip(keys, vals):
                    yield userids(u), itemids(k), v
            else:
                for k, v in zip(keys, vals):
                    yield u, k, v
            data_index = end

    def _iterate_stream(self, axis, use_repr_name, userids, itemids):
        assert axis in ['rowwise'], 'Unexpected date axis: {}'.format(axis)
        assert self.handle, 'Database is not opened'
        group = self.handle[axis]
        data_index = 0
        for u, end in enumerate(group['indptr']):
            keys = group['key'][data_index:end]
            if use_repr_name:
                for k in keys:
                    yield userids(u), itemids(k)
            else:
                for k in keys:
                    yield u, k
            data_index = end

    def iterate(self, axis='rowwise', use_repr_name=False) -> [int, int, float]:
        """Iterate over datas

        args:
            group: which data group
            use_repr_name: return representive name of internal ids
        """
        userids, itemids = None, None
        idmap = self.get_group('idmap')
        if use_repr_name:
            userids = lambda x: str(x)
            if idmap['rows'].shape[0] != 0:
                userids = lambda x: idmap['rows'][x].decode('utf-8', 'ignore')
            itemids = lambda x: str(x)
            if idmap['cols'].shape[0] != 0:
                itemids = lambda x: idmap['cols'][x].decode('utf-8', 'ignore')
            if axis == 'colwise':
                userids, itemids = itemids, userids

        if self.opt.data.internal_data_type == 'matrix':
            return self._iterate_matrix(axis, use_repr_name, userids, itemids)
        elif self.opt.data.internal_data_type == 'stream':
            return self._iterate_stream(axis, use_repr_name, userids, itemids)

    def get(self, index, axis='rowwise') -> [int, int, float]:
        if self.opt.data.internal_data_type == 'matrix':
            assert axis in ['rowwise', 'colwise'], 'Unexpected data axis: {}'.format(axis)
            assert self.handle, 'Database is not opened'
            group = self.handle[axis]
            begin = 0 if index == 0 else group['indptr'][index - 1]
            end = group['indptr'][index]
            keys = group['key'][begin:end]
            vals = group['val'][begin:end]
            return (keys, vals)
        elif self.opt.data.internal_data_type == 'stream':
            assert axis in ['rowwise'], 'Unexpected date axis: {}'.format(axis)
            assert self.handle, 'Database is not opened'
            group = self.handle[axis]
            begin = 0 if index == 0 else group['indptr'][index - 1]
            end = group['indptr'][index]
            keys = group['key'][begin:end]
            return (keys,)

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
        chunk_size = (min(1024 ** 3 - 1, num_nnz),)
        for g in [f['rowwise'], f['colwise']]:
            g.create_dataset('key', (num_nnz,), dtype='int32', maxshape=(num_nnz,), chunks=chunk_size)
            g.create_dataset('val', (num_nnz,), dtype='float32', maxshape=(num_nnz,), chunks=chunk_size)
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
        _, num_nnz = kwargs['num_users'], kwargs['num_nnz']
        method = self.opt.data.validation.name

        f.create_group('vali')
        g = f['vali']
        g.attrs['method'] = method
        g.attrs['n'] = 0
        if method == 'sample':
            sz = min(self.opt.data.validation.max_samples,
                     int(num_nnz * self.opt.data.validation.p))
            g.create_dataset('indexes', (sz,), dtype='int64', maxshape=(sz,))
            # Watch out, create_working_data cannot deal with last line of data
            # for validation data thus we have to reduce index 1.
            g['indexes'][:] = np.random.choice(num_nnz - 1, sz, replace=False)
            num_nnz -= sz
        elif method in ['newest']:
            sz = kwargs['num_validation_samples']
            g.attrs['n'] = self.opt.data.validation.n
            # We don't need to reduce sample size for validation samples. It
            # already applied on caller side.

        g.create_dataset('row', (sz,), dtype='int32', maxshape=(sz,))
        g.create_dataset('col', (sz,), dtype='int32', maxshape=(sz,))
        g.create_dataset('val', (sz,), dtype='float32', maxshape=(sz,))
        g.attrs['num_samples'] = sz
        return num_nnz

    def fill_validation_data(self, db, validation_data):
        if not validation_data:
            return
        validation_data = [line.strip().split() for line in validation_data]
        assert len(validation_data) == db['vali'].attrs['num_samples'], 'Mismatched validation data'

        num_users, num_items = db.attrs['num_users'], db.attrs['num_items']
        row = [int(r) - 1 for r, _, _ in validation_data]  # 0-based
        col = [int(c) - 1 for _, c, _ in validation_data]  # 0-based
        val = np.array([float(v) for _, _, v in validation_data], dtype=np.float32)
        temp_mat = csr_matrix((val, (row, col)), (num_users, num_items))
        db['vali']['row'][:] = row
        db['vali']['col'][:] = col
        db['vali']['val'][:] = self.value_prepro(temp_mat.data)

    def _prepare_validation_data(self):
        if hasattr(self, 'vali_data'):
            return True

        db = self.handle
        num_users, num_items = db.attrs['num_users'], db.attrs['num_items']
        row = db['vali']['row'][::]
        col = db['vali']['col'][::]
        val = db['vali']['val'][::]

        _temp_mat = csr_matrix((val, (row, col)), (num_users, num_items))
        indptr = _temp_mat.indptr[1:]
        key = _temp_mat.indices
        vali_rows = np.arange(len(indptr))[np.ediff1d(indptr, to_begin=indptr[0]) > 0]
        vali_gt = {
            u: set(key[indptr[u - 1]:indptr[u]]) if u != 0 else set(key[:indptr[0]])
            for u in vali_rows}
        validation_seen = {}
        max_seen_size = 0
        for rowid in vali_rows:
            seen, *_ = self.get(rowid)
            validation_seen[rowid] = set(seen)
            max_seen_size = max(len(seen), max_seen_size)
        validation_seen = validation_seen
        validation_max_seen_size = max_seen_size

        self.vali_data = {
            "row": row,
            "col": col,
            "val": val,
            "vali_rows": vali_rows,
            "vali_gt": vali_gt,
            "validation_seen": validation_seen,
            "validation_max_seen_size": validation_max_seen_size
        }
        return True

    def _sort_and_compressed_binarization(self, mm_path, num_lines, max_key, sort_key):
        num_workers = psutil.cpu_count()
        merged_bin = sort_and_compressed_binarization(
            mm_path,
            self.tmp_root,
            num_lines, max_key, sort_key, num_workers)
        return merged_bin

    def _load_compressed_triplet_bin(self, db, job_files, num_lines, max_key, is_colwise=0):
        self.logger.info('Load triplet files. Total job files: %s' % len(job_files))
        INDPTR_SIZE = 8
        RECORD_SIZE = 8
        record_estimated_size = num_lines * RECORD_SIZE
        indptr_estimated_size = max_key * INDPTR_SIZE
        indptr_file = job_files[0]
        job_files = job_files[1:]
        with open(indptr_file, 'rb') as fin:
            indptr_total_size = fin.seek(0, 2)
            fin.seek(0, 0)
            assert indptr_estimated_size == indptr_total_size, f'Not valid indptr file size {indptr_total_size} (excepted: {indptr_estimated_size})'
            indptr = np.frombuffer(fin.read(max_key * 8),
                                   dtype=np.int64,
                                   count=max_key)
            db['indptr'][:max_key] = indptr
        record_total_size = 0
        data_index = 0
        for job in job_files:
            with open(job, 'rb') as fin:
                total_size = fin.seek(0, 2)
                if total_size == 0:
                    continue
                record_total_size += total_size
                total_records = int(total_size / RECORD_SIZE)
                fin.seek(0, 0)
                data = np.frombuffer(fin.read(),
                                     dtype=np.dtype([('i', 'i'),
                                                     ('v', 'f')]),
                                     count=total_records)
                I, V = data['i'], data['v']
                if self.opt.data.value_prepro:
                    V = self.value_prepro(V.copy())
                db['key'][data_index:data_index + total_records] = I
                db['val'][data_index:data_index + total_records] = V
                data_index += total_records
        assert record_estimated_size == record_total_size, f'Not valid record file size {record_total_size} (excepted: {record_estimated_size})'
        os.remove(indptr_file)
        for path in job_files:
            os.remove(path)

    def _chunking_into_bins(self, mm_path, num_lines, max_key, sep_idx):
        num_workers = psutil.cpu_count()
        while num_workers > 20:
            num_workers = int(num_workers / 2)
        num_chunks = num_workers * 2
        self.logger.info(f'Dividing into {num_chunks} chunks...')
        job_files = chunking_into_bins(mm_path,
                                       self.tmp_root,
                                       total_lines=num_lines,
                                       num_chunks=num_chunks,
                                       sep_idx=sep_idx,
                                       num_workers=num_workers)
        return job_files

    def _build_compressed_triplets(self, db, job_files, num_lines, max_key, is_colwise=0):
        self.logger.info('Total job files: %s' % len(job_files))
        with log.ProgressBar(log.INFO, total=len(job_files), mininterval=10) as pbar:
            indptr_index = 0
            data_index = 0
            RECORD_SIZE = 12
            prev_key = 0
            for job in job_files:
                with open(job, 'rb') as fin:
                    total_size = fin.seek(0, 2)
                    if total_size == 0:
                        continue
                    total_records = int(total_size / RECORD_SIZE)
                    fin.seek(0, 0)
                    data = np.frombuffer(fin.read(),
                                         dtype=np.dtype([('u', 'i'),
                                                         ('i', 'i'),
                                                         ('v', 'f')]),
                                         count=total_records)
                    U, I, V = data['u'], data['i'], data['v']
                    if is_colwise:
                        U, I = I, U
                    if self.opt.data.value_prepro:
                        V = self.value_prepro(V.copy())
                    self.logger.debug("minU: {}, maxU: {}".format(U[0], U[-1]))
                    assert data_index + total_records <= num_lines, 'Requests data size(%s) exceed capacity(%s)' % (data_index + total_records, num_lines)
                    db['key'][data_index:data_index + total_records] = I
                    db['val'][data_index:data_index + total_records] = V
                    indptr = [data_index for j in range(U[0] - prev_key)]
                    indptr += [data_index + i
                               for i in range(1, total_records)
                               for j in range(U[i] - U[i - 1])]
                    db['indptr'][indptr_index:indptr_index + len(indptr)] = indptr
                    assert indptr_index + len(indptr) <= max_key
                    data_index += total_records
                    indptr_index += len(indptr)
                    prev_key = U[-1]
                pbar.update(1)
            db["indptr"][indptr_index:] = data_index
        for path in job_files:
            os.remove(path)

    def _build_data(self,
                    db,
                    working_data_path,
                    validation_data,
                    target_groups=['rowwise', 'colwise'],
                    sort=True):
        available_mb = psutil.virtual_memory().available / 1024 / 1024
        approximated_data_mb = 0
        with open(working_data_path, 'rb') as fin:
            fin.seek(0, 2)
            approximated_data_mb = db.attrs['num_nnz'] * 3 * 4 / 1024 / 1024
        buffer_mb = int(max(1024, available_mb * 0.75))
        # for each sides
        for group, sep_idx, max_key in [('rowwise', 0, db.attrs['num_users']),
                                        ('colwise', 1, db.attrs['num_items'])]:
            if group not in target_groups:
                continue
            self.logger.info(f'Building compressed triplets for {group}...')
            self.logger.info('Preprocessing...')
            self.prepro.pre(db)
            if approximated_data_mb * 1.2 < available_mb:
                self.logger.info('In-memory Compressing ...')
                job_files = self._sort_and_compressed_binarization(
                    working_data_path,
                    db.attrs['num_nnz'],
                    max_key,
                    sort_key=sep_idx + 1 if sort else -1)
                self._load_compressed_triplet_bin(
                    db[group], job_files,
                    num_lines=db.attrs['num_nnz'],
                    max_key=max_key,
                    is_colwise=sep_idx)
            else:
                self.logger.info('Disk-based Compressing...')
                if sort:
                    aux.psort(working_data_path,
                              tmp_dir=self.opt.data.tmp_dir,
                              key=sep_idx + 1,
                              buffer_mb=buffer_mb)
                job_files = self._chunking_into_bins(working_data_path,
                                                     db.attrs['num_nnz'],
                                                     max_key,
                                                     sep_idx=sep_idx)
                self._build_compressed_triplets(db[group],
                                                job_files,
                                                num_lines=db.attrs['num_nnz'],
                                                max_key=max_key,
                                                is_colwise=sep_idx)
            self.prepro.post(db[group])
            if group == 'rowwise':
                self.fill_validation_data(db, validation_data)
            self.logger.info('Finished')


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
