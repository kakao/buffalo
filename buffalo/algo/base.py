# -*- coding: utf-8 -*-
import os
import abc
import json
import tempfile

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
        if isinstance(opt_path, dict):
            opt_path = self.create_temporary_option_from_dict(opt_path)
        opt = aux.Option(opt_path)
        opt_path = opt_path
        self.is_valid_option(opt)
        return (opt, opt_path)


class AlgoOption(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_default_option(self) -> dict:
        pass

    @abc.abstractmethod
    def is_valid_option(self, opt) -> bool:
        pass

    def create_temporary_option_from_dict(self, opt) -> str:
        str_opt = json.dumps(opt)
        tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)
        tmp.write(str_opt)
        return tmp.name
