import os
import traceback

import h5py
import numpy as np
import scipy.io
import scipy.sparse

from buffalo.data import prepro
from buffalo.data.base import Data, DataOption, DataReader
from buffalo.misc import aux, log


class MatrixMarketOptions(DataOption):
    def get_default_option(self) -> aux.Option:
        opt = {
            "type": "matrix_market",
            "input": {
                "main": "",  # str or numpy-kind data
                "uid": "",  # if not set, row-id is used as userid. It is okay to pass list or 1d dence array as a id list information.
                "iid": ""  # if not set, col-id is used as itemid. It is okay to pass list or 1d dence array as a id list information.
            },
            "data": {
                "internal_data_type": "matrix",
                "validation": {
                    "name": "sample",
                    "p": 0.01,
                    "max_samples": 500
                },
                "batch_mb": 1024,
                "use_cache": False,
                "tmp_dir": "/tmp/",
                "path": "./mm.h5py",
                "disk_based": False  # use disk based data compressing
            }
        }
        return aux.Option(opt)

    def is_valid_option(self, opt) -> bool:
        assert super(MatrixMarketOptions, self).is_valid_option(opt)
        if not opt["type"] == "matrix_market":
            raise RuntimeError("Invalid data type: %s" % opt["type"])
        if opt["data"]["internal_data_type"] != "matrix":
            raise RuntimeError("MatrixMarket only support internal data type(matrix)")
        for field in ["uid", "iid"]:
            id_path = opt["input"][field]
            is_1d_dense = isinstance(id_path, (np.ndarray,)) and id_path.ndim == 1
            msg = f"Not supported data type for MatrixMarketOption.input.{field}: {type(id_path)}"
            assert any([isinstance(id_path, (str, list,)), is_1d_dense]), msg
        main = opt["input"]["main"]
        msg = f"Not supported data type for MatrixMarketOption.input.main field: {type(main)}"
        is_2d_dense = isinstance(main, (np.ndarray,)) and main.ndim == 2
        is_sparse = scipy.sparse.issparse(main)
        assert any([isinstance(main, (str,)), is_2d_dense, is_sparse]), msg
        return True


class MatrixMarketDataReader(DataReader):
    def __init__(self, opt, *args, **kwargs):
        super().__init__(opt, *args, **kwargs)

    def get_main_path(self):
        main = self.opt.input.main
        if isinstance(main, (str,)):
            return main

        if hasattr(self, "temp_main"):
            return self.temp_main

        log.get_logger("MatrixMarketDataReader").debug("creating temporary matrix-market data from numpy-kind array")
        tmp_path = aux.get_temporary_file(self.opt.data.tmp_dir)
        with open(tmp_path, "wb") as fout:
            if isinstance(main, (np.ndarray,)) and main.ndim == 2:
                main = scipy.sparse.csr_matrix(main)
            if scipy.sparse.issparse(main):
                scipy.io.mmwrite(fout, main)
                self.temp_main = tmp_path
                return tmp_path
        raise RuntimeError(f"Unexpected data type for MatrixMarketOption.input.main field: {type(main)}")

    def get_uid_path(self):
        uid = self.opt.input.uid
        if isinstance(uid, (str,)):
            return uid
        if uid is not None:
            return self._get_temporary_id_list_path(uid, "uid")
        return uid

    def get_iid_path(self):
        iid = self.opt.input.iid
        if isinstance(iid, (str,)):
            return iid
        if iid is not None:
            return self._get_temporary_id_list_path(iid, "iid")
        return iid


class MatrixMarket(Data):
    def __init__(self, opt, *args, **kwargs):
        super().__init__(opt, *args, **kwargs)
        self.name = "MatrixMarket"
        self.logger = log.get_logger("MatrixMarket")
        if isinstance(self.value_prepro,
                      (prepro.SPPMI)):
            raise RuntimeError(f"{self.opt.data.value_prepro.name} does not support MatrixMarket")
        self.data_type = "matrix"
        self.reader = MatrixMarketDataReader(self.opt)

    def _create(self, data_path, P, H):
        def get_max_column_length(fname):
            with open(fname) as fin:
                max_col = 0
                for l in fin:
                    max_col = max(max_col, len(l))
            return max_col

        uid_path, iid_path, main_path = P["uid_path"], P["iid_path"], P["main_path"]
        num_users, num_items, num_nnz = map(int, H.split())
        # Manually updating progress bar is a bit naive
        with log.ProgressBar(log.DEBUG, total=5, mininterval=30) as pbar:
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
                idmap = db["idmap"]
                # if not given, assume id as is
                if uid_path:
                    with open(uid_path) as fin:
                        idmap["rows"][:] = np.loadtxt(fin, dtype=f"S{uid_max_col}")
                else:
                    idmap["rows"][:] = np.array([str(i) for i in range(1, num_users + 1)],
                                                dtype=f"S{uid_max_col}")
                pbar.update(1)
                if iid_path:
                    with open(iid_path) as fin:
                        idmap["cols"][:] = np.loadtxt(fin, dtype=f"S{iid_max_col}")
                else:
                    idmap["cols"][:] = np.array([str(i) for i in range(1, num_items + 1)],
                                                dtype=f"S{iid_max_col}")
                pbar.update(1)
                num_header_lines = 0
                with open(main_path) as fin:
                    for line in fin:
                        if line.strip().startswith("%"):
                            num_header_lines += 1
                        else:
                            break
                pbar.update(1)
            except Exception as e:
                self.logger.error("Cannot create db: %s" % (str(e)))
                self.logger.error(traceback.format_exc())
                raise
        return db, num_header_lines

    def _create_working_data(self, db, source_path, ignore_lines):
        """
        Args:
            source_path: source data file path
            ignore_lines: number of lines to skip from start line
        """
        vali_indexes = [] if "vali" not in db else db["vali"]["indexes"]
        vali_lines = []
        file_path = aux.get_temporary_file(self.opt.data.tmp_dir)
        with open(file_path, "w") as w:
            fin = open(source_path, mode="r")
            file_size = fin.seek(0, 2)
            fin.seek(0, 0)
            for _ in range(ignore_lines):
                fin.readline()
            total = file_size - fin.tell()
            buffered = ""
            CHUNK_SIZE = 4096 * 1000
            total_lines = 0
            vali_indexes = sorted(vali_indexes)
            target_index = vali_indexes[0] if vali_indexes else -1
            vali_indexes = vali_indexes[1:]
            with log.ProgressBar(log.INFO, total=total, mininterval=10) as pbar:
                while True:
                    buffered += fin.read(CHUNK_SIZE)
                    if buffered == "":
                        break
                    current_file_position = fin.tell()
                    pbar.update(CHUNK_SIZE)
                    num_lines_on_buffer = buffered.count("\n")
                    # search the position of validation sample and extract
                    # it from training data
                    while target_index >= 0 and target_index < (total_lines + num_lines_on_buffer):
                        no_line = total_lines
                        new_buffered = ""
                        from_index = 0
                        for idx, c in enumerate(buffered):
                            if c == "\n":
                                if no_line == target_index:
                                    vali_lines.append(buffered[from_index:idx])
                                    if from_index > 0:
                                        w.write(buffered[0:from_index])
                                    new_buffered = buffered[idx + 1:]
                                    no_line += 1
                                    total_lines += 1
                                    num_lines_on_buffer -= 1
                                    break
                                no_line += 1
                                total_lines += 1
                                from_index = idx + 1
                                num_lines_on_buffer -= 1
                        buffered = new_buffered
                        if vali_indexes:
                            target_index, vali_indexes = vali_indexes[0], vali_indexes[1:]
                        else:
                            target_index = -1
                    where = buffered.rfind("\n")
                    total_lines += num_lines_on_buffer
                    if where != -1:
                        w.write(buffered[:where + 1])
                        buffered = buffered[where + 1:]
                    elif current_file_position == file_size:
                        w.write(buffered)
                        buffered = ""
            w.close()
            fin.close()
            return w.name, vali_lines

    def create(self) -> h5py.File:
        mm_main_path = self.reader.get_main_path()
        mm_uid_path = self.reader.get_uid_path()
        mm_iid_path = self.reader.get_iid_path()

        data_path = self.opt.data.path
        if os.path.isfile(data_path) and self.opt.data.use_cache:
            self.logger.info("Use cached DB on %s" % data_path)
            self.open(data_path)
            return

        self.logger.info("Create the database from matrix market file.")
        with open(mm_main_path) as fin:
            header = "%"
            while header.startswith("%"):
                header = fin.readline()
        self.logger.debug("Building meta part...")
        db, num_header_lines = self._create(data_path,
                                            {"main_path": mm_main_path,
                                                "uid_path": mm_uid_path,
                                                "iid_path": mm_iid_path},
                                            header)
        try:
            num_header_lines += 1  # add metaline
            self.logger.info("Creating working data...")
            tmp_main, validation_data = self._create_working_data(db,
                                                                  mm_main_path,
                                                                  num_header_lines)
            self.logger.debug(f"Working data is created on {tmp_main}")
            self.logger.info("Building data part...")
            self._build_data(db, tmp_main, validation_data)
            db.attrs["completed"] = 1
            db.close()
            self.handle = h5py.File(data_path, "r")
        except Exception as e:
            self.logger.error("Cannot create db: %s" % (str(e)))
            self.logger.error(traceback.format_exc().splitlines())
            if hasattr(self, "path"):
                if os.path.isfile(self.path):
                    os.remove(self.path)
            raise
        self.logger.info("DB built on %s" % data_path)
