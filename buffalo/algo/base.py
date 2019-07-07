# -*- coding: utf-8 -*-
import os
import abc

import numpy as np

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


class Serializable(abc.ABC):
    def dump(self, directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        meatadata_path = os.path.join(directory, 'metadata')
        with open(metadata_path, 'w') as fout:
            fout.write(json.dumps({'opt': self.opt}))

    def load(self, dir):
        pass
