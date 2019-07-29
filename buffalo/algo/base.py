# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import abc
import json
import pickle
import struct
import logging
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar
# what the...
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

from buffalo.misc import aux


class Algo(abc.ABC):
    def __init__(self, *args, **kwargs):
        self.temporary_files = []

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
    def __init__(self, *args, **kwargs):
        pass

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


class TensorboardExtention(object):
    @abc.abstractmethod
    def get_evaluation_metrics(self):
        raise NotImplemented

    def _get_initial_tensorboard_data(self):
        tb = aux.Option({'summary_writer': None,
                         'name': None,
                         'metrics': {},
                         'feed_dict': {},
                         'merged_summary_op': None,
                         'session': None,
                         'pbar': None,
                         'data_root': None,
                         'step': 1})
        return tb

    def initialize_tensorboard(self, num_steps, name_prefix='', name_postfix='', metrics=None):
        if not self.opt.tensorboard:
            if not hasattr(self, '_tb_setted'):
                self.logger.debug('Cannot find tensorboard configuration.')
            self.tb_setted = False
            return
        name = self.opt.tensorboard.name
        name = name_prefix + name + name_postfix
        dtm = datetime.datetime.now().strftime('%Y%m%d-%H.%M')
        template = self.opt.tensorboard.get('name_template', '{name}.{dtm}')
        self._tb = self._get_initial_tensorboard_data()
        self._tb.name = template.format(name=name, dtm=dtm)
        if not os.path.isdir(self.opt.tensorboard.root):
            os.makedirs(self.opt.tensorboard.root)
        tb_dir = os.path.join(self.opt.tensorboard.root, self._tb.name)
        self._tb.data_root = tb_dir
        self._tb.summary_writer = tf.summary.FileWriter(tb_dir)
        if not metrics:
            metrics = self.get_evaluation_metrics()
        for m in metrics:
            self._tb.metrics[m] = tf.placeholder(tf.float32)
            tf.summary.scalar(m, self._tb.metrics[m])
            self._tb.feed_dict[self._tb.metrics[m]] = 0.0
        self._tb.merged_summary_op = tf.summary.merge_all()
        self._tb.session = tf.Session()
        self._tb.pbar = Progbar(num_steps, stateful_metrics=self._tb.metrics)
        self._tb_setted = True

    def update_tensorboard_data(self, metrics):
        if not self.opt.tensorboard:
            return
        metrics = [(m, np.float32(metrics.get(m, 0.0)))
                   for m in self._tb.metrics.keys()]
        self._tb.feed_dict = {self._tb.metrics[k]: v
                              for k, v in metrics}
        summary = self._tb.session.run(self._tb.merged_summary_op,
                                       feed_dict=self._tb.feed_dict)
        self._tb.summary_writer.add_summary(summary, self._tb.step)
        self._tb.pbar.update(self._tb.step, metrics)
        self._tb.step += 1

    def finalize_tensorboard(self):
        if not self.opt.tensorboard:
            return
        with open(os.path.join(self._tb.data_root, 'opt.json'), 'w') as fout:
            fout.write(json.dumps(self.opt, indent=2))
        self._tb.summary_writer.close()
        self._tb.session.close()
        self._tb = None
        tf.reset_default_graph()

    def __del__(self):
        if hasattr(self, '_tb_setted') and self._tb_setted:
            tf.reset_default_graph()
