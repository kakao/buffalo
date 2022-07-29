# -*- coding: utf-8 -*-
"""Buffalo
"""
DOCLINES = __doc__.split("\n")

import os
import sys
import platform
import sysconfig
import subprocess

import numpy as np
from setuptools import setup, Extension

from cuda_setup import CUDA, build_ext


# TODO: Python3 Support
if sys.version_info[:3] < (3, 6):
    raise RuntimeError("Python version 3.6 or later required.")

assert platform.system() == 'Linux'  # TODO: MacOS

MAJOR = 1
MINOR = 2
MICRO = 2
Release = True
STAGE = {True: '', False: 'b'}.get(Release)
VERSION = f'{MAJOR}.{MINOR}.{MICRO}{STAGE}'
STATUS = {False: 'Development Status :: 4 - Beta',
          True: 'Development Status :: 5 - Production/Stable'}

CLASSIFIERS = """{status}
Programming Language :: C++
Programming Language :: Cython
Programming Language :: Python :: 3.6
Operating System :: POSIX :: Linux
Operating System :: Unix
Operating System :: MacOS
License :: OSI Approved :: Apache Software License""".format(status=STATUS.get(Release))

CLIB_DIR = os.path.join(sysconfig.get_path('purelib'), 'buffalo')
numpy_include_dirs = np.get_include()
LIBRARY_DIRS = [CLIB_DIR]
EXTRA_INCLUDE_DIRS = [numpy_include_dirs,
                      '3rd/json11',
                      '3rd/spdlog/include',
                      '3rd/eigen3']


def get_extend_compile_flags():
    flags = ['-march=native']
    return flags


class CMakeExtension(Extension, object):
    extension_type = 'cmake'

    def __init__(self, name, **kwargs):
        super(CMakeExtension, self).__init__(name, sources=[])


extend_compile_flags = get_extend_compile_flags()

extensions = [
    CMakeExtension(name="cbuffalo"),
    Extension(name="buffalo.algo._als",
              sources=['buffalo/algo/_als.pyx'],
              language='c++',
              include_dirs=['./include'] + EXTRA_INCLUDE_DIRS,
              libraries=['gomp', 'cbuffalo'],
              library_dirs=LIBRARY_DIRS,
              runtime_library_dirs=LIBRARY_DIRS,
              extra_compile_args=['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + extend_compile_flags),
    Extension(name="buffalo.algo._cfr",
              sources=['buffalo/algo/_cfr.pyx'],
              language='c++',
              include_dirs=['./include'] + EXTRA_INCLUDE_DIRS,
              libraries=['gomp', 'cbuffalo'],
              library_dirs=LIBRARY_DIRS,
              runtime_library_dirs=LIBRARY_DIRS,
              extra_compile_args=['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + extend_compile_flags),
    Extension(name="buffalo.algo._bpr",
              sources=['buffalo/algo/_bpr.pyx'],
              language='c++',
              include_dirs=['./include'] + EXTRA_INCLUDE_DIRS,
              libraries=['gomp', 'cbuffalo'],
              library_dirs=LIBRARY_DIRS,
              runtime_library_dirs=LIBRARY_DIRS,
              extra_compile_args=['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + extend_compile_flags),
    Extension(name="buffalo.algo._plsi",
              sources=['buffalo/algo/_plsi.pyx'],
              language='c++',
              include_dirs=['./include'] + EXTRA_INCLUDE_DIRS,
              libraries=['gomp', 'cbuffalo'],
              library_dirs=LIBRARY_DIRS,
              runtime_library_dirs=LIBRARY_DIRS,
              extra_compile_args=['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + extend_compile_flags),
    Extension(name="buffalo.algo._warp",
              sources=['buffalo/algo/_warp.pyx'],
              language='c++',
              include_dirs=['./include'] + EXTRA_INCLUDE_DIRS,
              libraries=['gomp', 'cbuffalo'],
              library_dirs=LIBRARY_DIRS,
              runtime_library_dirs=LIBRARY_DIRS,
              extra_compile_args=['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + extend_compile_flags),
    Extension(name="buffalo.algo._w2v",
              sources=['buffalo/algo/_w2v.pyx'],
              language='c++',
              include_dirs=['./include'] + EXTRA_INCLUDE_DIRS,
              libraries=['gomp', 'cbuffalo'],
              library_dirs=LIBRARY_DIRS,
              runtime_library_dirs=LIBRARY_DIRS,
              extra_compile_args=['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + extend_compile_flags),
    Extension(name="buffalo.misc._log",
              sources=['buffalo/misc/_log.pyx'],
              language='c++',
              include_dirs=['./include'] + EXTRA_INCLUDE_DIRS,
              libraries=['gomp', 'cbuffalo'],
              library_dirs=LIBRARY_DIRS,
              runtime_library_dirs=LIBRARY_DIRS,
              extra_compile_args=['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + extend_compile_flags),
    Extension(name="buffalo.data.fileio",
              sources=['buffalo/data/fileio.pyx'],
              language='c++',
              libraries=['gomp'],
              extra_compile_args=['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + extend_compile_flags),
    Extension(name="buffalo.parallel._core",
              sources=['buffalo/parallel/_core.pyx'],
              language='c++',
              libraries=['gomp', 'n2'],
              include_dirs=EXTRA_INCLUDE_DIRS + ['./3rd/n2/include', './3rd/'],
              library_dirs=LIBRARY_DIRS,
              runtime_library_dirs=LIBRARY_DIRS,
              extra_compile_args=['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + extend_compile_flags),
]

if CUDA:
    extra_compile_args = ['-std=c++14', '-ggdb', '-O3'] + extend_compile_flags
    extensions.append(Extension("buffalo.algo.cuda._als",
                                sources=["buffalo/algo/cuda/_als.pyx",
                                         "lib/cuda/als/als.cu",
                                         "./3rd/json11/json11.cpp",
                                         "lib/misc/log.cc"],
                                language="c++",
                                extra_compile_args=extra_compile_args,
                                library_dirs=[CUDA['lib64']],
                                libraries=['cudart', 'cublas', 'curand'],
                                include_dirs=["./include", numpy_include_dirs,
                                              CUDA['include'], "./3rd/json11",
                                              "./3rd/spdlog/include"]))
    extensions.append(Extension("buffalo.algo.cuda._bpr",
                                sources=["buffalo/algo/cuda/_bpr.pyx",
                                         "lib/cuda/bpr/bpr.cu",
                                         "./3rd/json11/json11.cpp",
                                         "lib/misc/log.cc"],
                                language="c++",
                                extra_compile_args=extra_compile_args,
                                library_dirs=[CUDA['lib64']],
                                libraries=['cudart', 'cublas', 'curand'],
                                include_dirs=["./include", numpy_include_dirs,
                                              CUDA['include'], "./3rd/json11",
                                              "./3rd/spdlog/include"]))
else:
    print("Failed to find CUDA toolkit. Building without GPU acceleration.")


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def write_version_py(filename='buffalo/version.py'):
    cnt = """
short_version = '%(version)s'
git_revision = '%(git_revision)s'
"""
    GIT_REVISION = git_version()
    with open(filename, 'w') as fout:
        fout.write(cnt % {'version': VERSION,
                          'git_revision': GIT_REVISION})


class BuildExtension(build_ext, object):
    def run(self):
        for ext in self.extensions:
            if hasattr(ext, 'extension_type') and ext.extension_type == 'cmake':
                self.cmake(ext)
        super(BuildExtension, self).run()

    def cmake(self, ext):
        cwd = os.path.abspath(os.getcwd())
        os.makedirs(self.build_temp, exist_ok=True)

        build_type = 'Debug' if self.debug else 'Release'

        cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + build_type,
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + CLIB_DIR,
        ]

        build_args = []

        os.chdir(self.build_temp)
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        os.chdir(str(cwd))


def setup_package():
    write_version_py()
    cmdclass = {
        'build_ext': BuildExtension
    }

    with open('requirements.txt', 'r') as fin:
        install_requires = [line.strip() for line in fin]

    metadata = dict(
        name='buffalo',
        maintainer="lucas kwangseob kim",
        maintainer_email="recoteam@kakaocorp.com",
        description=DOCLINES[0],
        long_description="\n".join(DOCLINES[2:]),
        url="https://github.com/kakao/buffalo",
        download_url="https://github.com/kakao/buffalo/releases",
        include_package_data=False,
        license='Apache2',
        packages=['buffalo/algo/',
                  'buffalo/algo/cuda',
                  'buffalo/data/',
                  'buffalo/evaluate/',
                  'buffalo/parallel/',
                  'buffalo/misc/',
                  'buffalo/'],
        cmdclass=cmdclass,
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms=['Linux', 'Mac OSX', 'Unix'],
        ext_modules=extensions,
        entry_points={
            'console_scripts': [
                'Buffalo = buffalo.cli:_cli_buffalo',
            ]
        },
        python_requires='>=3.6',
        install_requires=install_requires,
    )

    metadata['version'] = VERSION
    setup(**metadata)


if __name__ == '__main__':
    setup_package()
