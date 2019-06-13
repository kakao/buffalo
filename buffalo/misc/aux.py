# -*- coding: utf-8 -*-
import os


def get_logger(name=__file__):
    import logging
    import logging.handlers
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)-8s] %(asctime)s [%(filename)s] [%(funcName)s:%(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S')
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    return logger


class Option(dict):
    def __init__(self, *args, **kwargs):
        import json
        args = [arg if isinstance(arg, dict) else json.loads(open(arg).read())
                for arg in args]
        super(Option, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        self[k] = Option(v)
                    else:
                        self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    self[k] = Option(v)
                else:
                    self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Option, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Option, self).__delitem__(key)
        del self.__dict__[key]


def mkdirs(path):
    if not path:
        return True
    if os.path.isdir(path):
        return True
    if path.endswith('/'):
        path = path[:-1]
    parent = path.split('/')[:-1]
    mkdirs('/'.join(parent))
    os.mkdir(path)
