# -*- coding: utf-8 -*-
import os
import abc
import json
import atexit
import psutil
import warnings
import tempfile
import subprocess

from buffalo.misc import log

_temporary_files = []


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

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)


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
            _temporary_files.append(tmp.name)
            return tmp.name


def copy_to_temporary_file(source_path, ignore_lines=0, chunk_size=8192, binary=False):
    W = 'w' if not binary else 'wb'
    R = 'r' if not binary else 'rb'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        with tempfile.NamedTemporaryFile(mode=W, delete=False) as w:
            fin = open(source_path, mode=R)
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


def psort(path, parallel=-1, field_seperator=' ', key=1, tmp_dir='/tmp/', buffer_mb=1024, output=None):
    # TODO: We need better way for OS/platform compatibility.
    # we need compatibility checking routine for this method.
    commands = ['sort', '-n', '-s']
    if parallel == -1:
        parallel = psutil.cpu_count()
    if parallel > 0:
        commands.extend(['--parallel', parallel])
    if not output:
        output = path
    commands.extend(['-t', '{}'.format(field_seperator)])
    commands.extend(['-k', key])
    commands.extend(['-T', tmp_dir])
    commands.extend(['-S', '%sM' % buffer_mb])
    commands.extend(['-o', output])
    commands.append(path)
    try:
        subprocess.check_output(map(str, commands), stderr=subprocess.STDOUT, env={'LC_ALL': 'C'})
    except Exception as e:
        log.get_logger().error('Unexpected error: %s for %s' % (str(e), ' '.join(list(map(str, commands)))))
        raise


def get_temporary_file(root='/tmp/', write_mode='w'):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        w = tempfile.NamedTemporaryFile(mode=write_mode, dir=root, delete=False)
        _temporary_files.append(w.name)
        return w.name


@atexit.register
def __cleanup_tempory_files():
    for path in _temporary_files:
        if os.path.isfile(path):
            os.remove(path)


def register_cleanup_file(path):
    _temporary_files.append(path)
