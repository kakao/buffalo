# -*- coding: utf-8 -*-
import os
import abc
import pickle
import struct

import numpy as np

from buffalo.misc import aux


class Algo(abc.ABC):
    def __init__(self, *args, **kwargs):
        self.temporary_files = []

    @abc.abstractmethod
    def init(self):
        raise NotImplemented

    def __del__(self):
        for path in self.temporary_files:
            try:
                os.remove(path)
            except Exception as e:
                self.logger.warning('Cannot remove temporary file: %s, Error: %s' % (path, e))

    def get_option(self, opt_path) -> (aux.Option, str):
        if isinstance(opt_path, (dict, aux.Option)):
            opt_path = self.create_temporary_option_from_dict(opt_path)
            self.temporary_files.append(opt_path)
        opt = aux.Option(opt_path)
        opt_path = opt_path
        self.is_valid_option(opt)
        return (aux.Option(opt), opt_path)

    def normalize(self, feat):
        feat = feat / np.sqrt((feat ** 2).sum(-1) + 1e-8)[..., np.newaxis]
        return feat

    def topk_recommendation(self, keys, topk, encode='utf-8') -> dict:
        """Return TopK recommendation for each users(keys)
        """
        if not isinstance(keys, list):
            keys = [keys]
        if not self.data.id_mapped:
            self.logger.warning('If DataID is not mapped, the results may be empty')
        rows = [self.data.userid_map[k] for k in keys if k in self.data.userid_map]
        topks = self._get_topk_recommendation(rows, topk)
        idmap = self.data.get_group('idmap')
        if not encode:
            return {idmap['rows'][k]: [idmap['cols'][v] for v in vv]
                    for k, vv in topks}
        else:
            return {idmap['rows'][k].decode(encode, 'ignore'): [idmap['cols'][v].decode(encode, 'ignore') for v in vv]
                    for k, vv in topks}

    def most_similar(self, keys, topk, encode='utf-8') -> dict:
        """Return Most similar items for each items(keys)
        """
        if not isinstance(keys, list):
            keys = [keys]
        if not self.data.id_mapped:
            self.logger.warning('If DataID is not mapped, the results may be empty')
        cols = [self.data.itemid_map[k] for k in keys if k in self.data.itemid_map]
        topks = self._get_most_similar(cols, topk)
        idmap = self.data.get_group('idmap')
        if not encode:
            return {idmap['cols'][k]: [idmap['cols'][v] for v in vv]
                    for k, vv in topks}
        else:
            return {idmap['cols'][k].decode(encode, 'ignore'): [idmap['cols'][v].decode(encode, 'ignore') for v in vv]
                    for k, vv in topks}


class Serializable(abc.ABC):
    def dump(self, path):
        data = self._get_data()
        with open(path, 'wb') as fout:
            total_objs = len(data)
            fout.write(struct.pack('Q', total_objs))
            for name, obj in data:
                bname = bytes(name, encoding='utf-8')
                fout.write(struct.pack('Q', len(bname)))
                fout.write(bname)
                s = pickle.dumps(obj, protocol=4)
                fout.write(struct.pack('Q', len(s)))
                fout.write(s)

    @abc.abstractmethod
    def _get_data(self):
        raise NotImplemented

    def load(self, path):
        with open(path, 'rb') as fin:
            total_objs = struct.unpack('Q', fin.read(8))[0]
            for _ in range(total_objs):
                name_sz = struct.unpack('Q', fin.read(8))[0]
                name = fin.read(name_sz).decode('utf8')
                obj_sz = struct.unpack('Q', fin.read(8))[0]
                obj = pickle.loads(fin.read(obj_sz))
                setattr(self, name, obj)
