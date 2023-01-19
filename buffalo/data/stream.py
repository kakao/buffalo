import os
import traceback
import warnings
from collections import Counter

import h5py
import numpy as np
import psutil

from buffalo.data.base import Data, DataOption
from buffalo.data.fileio import parallel_build_sppmi
from buffalo.misc import aux, log


class StreamOptions(DataOption):
    """
    The option class for Stream class
    Options:
        type: Must be "stream"
        input:
            main: Path to main stream file.
            uid: User names corresponding to each line.
            iid: Item names corresponding to each item id.
        data:
            validation: See validation section.
            batch_mb: Internal batch size. Generally, the larger size, faster.
            use_cache: Set True to use already built data, otherwise building new one every time.
            disk_based: Set True to use disk based data compressing with low memory usage.
              Otherwise, base on approximated data size system will decided on its own.
            tmp_dir: Where temporary files goes on.
            path: Output path of Stream.
            internal_data_type: "stream" or "matrix"
                stream: Data file treated as is(i.e. streaming data)
                matrix: Translate data file into sparse matrix format(it lose sequential information, but more compact for data size).
            sppmi:
                windows: window size to set relation between word and context
                k: sppmi shift k (serves as negative sampling size in w2v model)
    """
    def get_default_option(self) -> aux.Option:
        opt = {
            "type": "stream",
            "input": {
                "main": "",
                "uid": "",  # if not set, row-id is used as userid.
                "iid": ""  # if not set, col-id is used as userid.
            },
            "data": {
                "validation": {
                    "name": "newest",  # sample or newest
                    "p": 0.01,  # if set newest, ignored
                    "n": 1,  # if set sample, ignored
                    "max_samples": 500
                },
                "sppmi": {
                    # "windows": 5,
                    # "k": 1
                },
                "batch_mb": 1024,
                "use_cache": False,
                "tmp_dir": "/tmp/",
                "path": "./stream.h5py",
                "internal_data_type": "stream",  # if set to "matrix", internal data stored as like matrix market format
                "disk_based": False
            }
        }
        return aux.Option(opt)

    def is_valid_option(self, opt) -> bool:
        assert super(StreamOptions, self).is_valid_option(opt)
        if not opt["type"] == "stream":
            raise RuntimeError("Invalid data type: %s" % opt["type"])
        return True


class Stream(Data):
    def __init__(self, opt, *args, **kwargs):
        super(Stream, self).__init__(opt, *args, **kwargs)
        self.name = "Stream"
        self.logger = log.get_logger("Stream")
        self.data_type = "stream"

    def _create(self, data_path, P):
        def get_max_column_length(fname):
            with open(fname) as fin:
                max_col = 0
                for l in fin:
                    max_col = max(max_col, len(l))
            return max_col
        uid_path, iid_path, main_path = P["uid_path"], P["iid_path"], P["main_path"]
        if uid_path:
            with open(uid_path) as fin:
                num_users = len([1 for _ in fin])
        else:
            with open(main_path) as fin:
                num_users = len([1 for _ in fin])

        uid_max_col = len(str(num_users)) + 1
        if uid_path:
            uid_max_col = get_max_column_length(uid_path) + 1

        vali_n = self.opt.data.validation.get("n", 0)
        num_nnz, vali_limit, itemids = 0, 0, set()
        self.logger.info(f"gathering itemids from {main_path}...")
        if self.opt.data.validation.name not in ["newest"]:
            vali_n = 0
        with open(main_path) as fin:
            for line in log.ProgressBar(level=log.DEBUG, iterable=fin):
                data = line.strip().split()
                if not iid_path:
                    itemids |= set(data)

                data_size = len(data)
                _vali_size = min(vali_n, len(data) - 1)
                vali_limit += _vali_size
                if self.opt.data.internal_data_type == "stream":
                    num_nnz += (data_size - _vali_size)
                elif self.opt.data.internal_data_type == "matrix":
                    num_nnz += len(set(data[:(data_size - _vali_size)]))
        if iid_path:
            with open(iid_path) as fin:
                itemids = {iid.strip(): idx + 1 for idx, iid in enumerate(fin)}
        else:  # in case of item information is not given
            itemids = {i: idx + 1 for idx, i in enumerate(itemids)}
        iid_max_col = max(len(k) + 1 for k in itemids.keys())
        num_items = len(itemids)

        self.logger.info("Found %d unique itemids" % len(itemids))

        try:
            db = self._create_database(data_path,
                                       num_users=num_users,
                                       num_items=num_items,
                                       num_nnz=num_nnz,
                                       uid_max_col=uid_max_col,
                                       iid_max_col=iid_max_col,
                                       num_validation_samples=vali_limit)
            idmap = db["idmap"]
            # if not given, assume id as is
            if uid_path:
                with open(uid_path) as fin:
                    idmap["rows"][:] = np.loadtxt(fin, dtype=f"S{uid_max_col}")
            else:
                idmap["rows"][:] = np.array([str(i) for i in range(1, num_users + 1)],
                                            dtype=f"S{uid_max_col}")
            if iid_path:
                with open(iid_path) as fin:
                    idmap["cols"][:] = np.loadtxt(fin, dtype=f"S{iid_max_col}")
            else:
                cols = sorted(itemids.items(), key=lambda x: x[1])
                cols = [k for k, _ in cols]
                idmap["cols"][:] = np.array(cols, dtype=f"S{iid_max_col}")
        except Exception as e:
            self.logger.error("Cannot create db: %s" % (str(e)))
            self.logger.error(traceback.format_exc())
            raise
        return db, itemids

    def _build_data(self, db, working_data_path, validation_data):
        if self.opt.data.internal_data_type == "stream":
            super()._build_data(db, working_data_path, validation_data,
                                target_groups=["rowwise"],
                                sort=False)  # keep order
        elif self.opt.data.internal_data_type == "matrix":
            super()._build_data(db, working_data_path, validation_data,
                                target_groups=["rowwise", "colwise"])

    def _build_sppmi(self, db, working_data_path, sppmi_total_lines, k):
        self.logger.info(f"build sppmi (shift k: {k})")
        sz = db.attrs["num_items"]
        nnz = 0
        self.logger.debug("sort working_data")
        aux.psort(working_data_path, key=1)
        w_path = aux.get_temporary_file(root=self.opt.data.tmp_dir)
        self.logger.debug(f"build sppmi in_parallel. w: {w_path}")
        num_workers = psutil.cpu_count()
        nnz = parallel_build_sppmi(working_data_path, w_path, sppmi_total_lines, sz, k, num_workers)
        self.logger.debug(f"sort output. w: {w_path}, nnz: {nnz}")
        aux.psort(w_path)
        self.logger.info(f"convert from {working_data_path} to {w_path}")
        db.create_group("sppmi")
        db.attrs["sppmi_nnz"] = nnz
        self.logger.info(f"sppmi nnz: {nnz}")
        db["sppmi"].create_dataset("indptr", (sz,), dtype="int64", maxshape=(sz,))
        db["sppmi"].create_dataset("key", (nnz,), dtype="int32", maxshape=(nnz,))
        db["sppmi"].create_dataset("val", (nnz,), dtype="float32", maxshape=(nnz,))
        self.logger.info("Disk-based Compressing...")
        job_files = self._chunking_into_bins(w_path, nnz, sz, 0)
        self._build_compressed_triplets(db["sppmi"],
                                        job_files,
                                        num_lines=nnz,
                                        max_key=sz,
                                        is_colwise=0)

    def _create_working_data(self, db, stream_main_path, itemids,
                             with_sppmi=False, windows=5):
        vali_method = None if "vali" not in db else db["vali"].attrs["method"]
        vali_indexes, vali_n = set(), 0
        if vali_method == "sample":
            vali_indexes = set(db["vali"]["indexes"])
        elif vali_method in ["newest"]:
            vali_n = db["vali"].attrs["n"]
        vali_lines = []
        # users = db["idmap"]["rows"][:] will be used someday?
        sppmi_total_lines = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            if with_sppmi:
                w_sppmi = open(aux.get_temporary_file(root=self.opt.data.tmp_dir), "w")
            file_path = aux.get_temporary_file(root=self.opt.data.tmp_dir)
            with open(stream_main_path) as fin, open(file_path, "w") as w:
                total_index = 0
                internal_data_type = self.opt.data.internal_data_type
                for line_idx, data in log.ProgressBar(level=log.DEBUG, iterable=enumerate(fin)):
                    data = data.strip().split()
                    # total_data_size = len(data) will be used someday?
                    user = line_idx + 1
                    vali_data, train_data = [], []
                    if vali_method in ["newest"]:
                        vali_data_size = min(vali_n, len(data) - 1)
                        train_data_size = len(data) - vali_data_size
                        vali = data[train_data_size:]
                        data = data[:train_data_size]
                        for col, val in Counter(vali).items():
                            col = itemids[col]
                            vali_data.append(col)
                    if internal_data_type == "stream":
                        for idx, col in enumerate(data):
                            col = itemids[col]
                            if (idx + total_index) in vali_indexes:
                                vali_data.append(col)
                            else:
                                train_data.append(col)
                    elif internal_data_type == "matrix":
                        for idx, col in enumerate(data):
                            col = itemids[col]
                            if (idx + total_index) in vali_indexes:
                                vali_data.append(col)
                            else:
                                train_data.append(col)
                    total_index += len(data)
                    if internal_data_type == "stream":
                        for col in train_data:
                            w.write(f"{user} {col} 1\n")
                        for col in vali_data:
                            vali_lines.append(f"{user} {col} {val}")
                    else:
                        for col, val in Counter(train_data).items():
                            w.write(f"{user} {col} {val}\n")
                        for col, val in Counter(vali_data).items():
                            vali_lines.append(f"{user} {col} {val}")
                    if with_sppmi:
                        sz = len(train_data)
                        for i in range(sz):
                            beg, end = i + 1, i + windows + 1
                            for j in range(beg, end):
                                if j >= sz:
                                    break
                                _w, _c = train_data[i], train_data[j]
                                w_sppmi.write(f"{_w} {_c}\n")
                                w_sppmi.write(f"{_c} {_w}\n")
                                sppmi_total_lines += 2
                if with_sppmi:
                    w_sppmi.close()
                    return w.name, vali_lines, w_sppmi.name, sppmi_total_lines
                return w.name, vali_lines, None, None

    def create(self) -> h5py.File:
        stream_main_path = self.opt.input.main
        stream_uid_path = self.opt.input.uid
        stream_iid_path = self.opt.input.iid

        data_path = self.opt.data.path
        if os.path.isfile(data_path) and self.opt.data.use_cache:
            self.logger.info("Use cached DB on %s" % data_path)
            self.open(data_path)
            return

        self.logger.info("Create database from stream data")

        self.logger.debug("Building meta part...")
        db, itemids = self._create(data_path,
                                   {"main_path": stream_main_path,
                                    "uid_path": stream_uid_path,
                                    "iid_path": stream_iid_path})
        sppmi_opt = self.opt.data.sppmi
        try:
            self.logger.info("Creating working data...")
            if sppmi_opt:
                tmp_main, validation_data, tmp_sppmi, sppmi_total_lines = \
                    self._create_working_data(db, stream_main_path, itemids, True, sppmi_opt.windows)
            else:
                tmp_main, validation_data, tmp_sppmi, sppmi_total_lines = \
                    self._create_working_data(db, stream_main_path, itemids, False)
            self.logger.debug(f"Working data is created on {tmp_main}")
            self.logger.info("Building data part...")
            self._build_data(db, tmp_main, validation_data)
            if sppmi_opt:
                self.logger.debug(f"sppmi data is created on {tmp_sppmi}")
                self._build_sppmi(db, tmp_sppmi, sppmi_total_lines, sppmi_opt.k)
            db.attrs["completed"] = 1
            db.close()
            self.handle = h5py.File(data_path, "r")
        except Exception as e:
            self.logger.error("Cannot create db: %s" % (str(e)))
            self.logger.error(traceback.format_exc())
            if hasattr(self, "path"):
                if os.path.isfile(self.path):
                    os.remove(self.path)
            raise
        self.logger.info("DB built on %s" % data_path)
