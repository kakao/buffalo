# -*- coding: utf-8 -*-
import os
import abc
import json
import psutil
import warnings
import tempfile
import subprocess


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


class InputOptions(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_default_option(self) -> dict:
        pass

    def is_valid_option(self, opt) -> bool:
        default_opt = self.get_default_option()
        keys = self.get_default_option()
        for key in keys:
            if key not in opt:
                raise RuntimeError('{} not exists on Option'.format(key))
            expected_type = type(default_opt[key])
            if not isinstance(opt.get(key), expected_type):
                raise RuntimeError('Invalid type for {}, {} expected. '.format(key, type(default_opt[key])))
        return True

    def create_temporary_option_from_dict(self, opt) -> str:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            str_opt = json.dumps(opt)
            tmp = tempfile.NamedTemporaryFile(mode='w', dir=opt.get('tmp_dir', '/tmp/'), delete=False)
            tmp.write(str_opt)
            return tmp.name


def make_temporary_file(path, ignore_lines=0, chunk_size=8192, binary=False):
    W = 'w' if not binary else 'wb'
    R = 'r' if not binary else 'rb'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        with tempfile.NamedTemporaryFile(mode=W, delete=False) as w:
            fin = open(path, mode=R)
            for _ in range(ignore_lines):
                fin.readline()
            while True:
                chunk = fin.read(chunk_size)
                if chunk:
                    w.write(chunk)
                if len(chunk) != chunk_size:
                    break
            w.close()
            return w.name


def psort(path, parallel=-1, field_seperator=' ', key=1, output=None):
    commands = ['sort', '-n']
    if parallel == -1:
        parallel = psutil.cpu_count()
    if parallel > 0:
        commands.extend(['--parallel', parallel])
    if not output:
        output = path
    commands.extend(['-t', '{}'.format(field_seperator)])
    commands.extend(['-k', key])
    commands.extend(['-o', output])
    commands.append(path)
    subprocess.check_output(map(str, commands), stderr=subprocess.STDOUT)
