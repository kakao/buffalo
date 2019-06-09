# -*- coding: utf-8 -*-

"""Buffalo
"""
DOCLINES = __doc__.split("\n")

import os
import sys
import pathlib
import platform
import subprocess
from setuptools import setup
from ConfigParser import ConfigParser
from distutils.extension import Extension
from setuptools.command.build_ext import build_ext
try:
    from sphinx.setup_command import BuildDoc
    HAVE_SPHINX = True
except:
    HAVE_SPHINX = False

import numpy
import eigency

# TODO: Python3 Support
if sys.version_info[:2] < (2, 7) or (2, 8) <= sys.version_info[0:2]:
    raise RuntimeError("Python version 2.7 required.")
assert platform.system() == 'Linux'  # TODO: MacOS
numpy_include_dirs = os.path.split(numpy.__file__)[0] + '/core/include'

CLASSIFIERS = """Development Status :: 0.1.0 - Dev
Programming Language :: C/C++
Programming Language :: Cythonj
Programming Language :: Python :: 2.7
Operating System :: Unix
Operating System :: MacOS"""

MAJOR = 0
MINOR = 1
MICRO = 0
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
site_cfg = ConfigParser()
site_cfg.read('site.cfg')


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
    Extension(name="buffalo.algo.als",
              sources=['buffalo/algo/als.cpp'],
              include_dirs=['./include',
                            numpy_include_dirs,
                            '3rd/json11',
                            '3rd/spdlog/include',
                            site_cfg.get('eigen', 'include_dirs')] + eigency.get_includes(),
              libraries=['gomp', 'cbuffalo'],
              library_dirs=['/usr/local/lib64'],
              extra_compile_args=['-fopenmp', '-std=c++14', '-O3'] + extend_compile_flags),
    Extension(name="buffalo.misc.log",
              sources=['buffalo/misc/log.cpp'],
              include_dirs=['./include',
                            numpy_include_dirs,
                            '3rd/json11',
                            '3rd/spdlog/include',
                            site_cfg.get('eigen', 'include_dirs')] + eigency.get_includes(),
              libraries=['gomp', 'cbuffalo'],
              library_dirs=['/usr/local/lib64'],
              extra_compile_args=['-fopenmp', '-std=c++14', '-O3'] + extend_compile_flags),
]


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


class BuildExtention(build_ext, object):
    def run(self):
        for ext in self.extensions:
            if hasattr(ext, 'extension_type') and ext.extension_type == 'cmake':
                self.cmake(ext)
        self.cythonize()
        super(BuildExtention, self).run()

    def cythonize(self):
        ext_files = ['buffalo/algo/als.pyx',
                     'buffalo/misc/log.pyx']
        for path in ext_files:
            from Cython.Build import cythonize
            cythonize(path)

    def cmake(self, ext):
        cwd = pathlib.Path().absolute()

        build_temp = pathlib.Path(self.build_temp)
        try:
            build_temp.mkdir(parents=True)
        except OSError:
            pass
        libdir = pathlib.Path(self.build_lib)
        try:
            libdir.mkdir(parents=True)
        except OSError:
            pass
        build_type = 'Debug' if self.debug else 'Release'

        cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + build_type,
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(libdir.absolute()),
        ]

        build_args = []

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args + ['--target', 'install'])
        os.chdir(str(cwd))


def setup_package():
    write_version_py()
    cmdclass = {'build_ext': BuildExtention}
    if HAVE_SPHINX:
        cmdclass.update({'build_sphinx': BuildDoc})

    build_requires = [l.strip() for l in open('requirements.txt')]

    metadata = dict(
        name='buffalo',
        maintainer="lucas kwangseob kim",
        maintainer_email="recoteam@kakaocorp.com",
        description=DOCLINES[0],
        long_description="\n".join(DOCLINES[2:]),
        url="https://github.daumkakao.com/toros/buffalo",
        download_url="https://github.daumkakao.com/toros/buffalo/releases",
        include_package_data=False,
        license='Apache2',
        packages=['buffalo/algo/',
                  'buffalo/data/',
                  'buffalo/evaluate/',
                  'buffalo/misc/',
                  'buffalo/util/',
                  'buffalo/'],
        cmdclass=cmdclass,
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms=['Linux', 'Mac OSX', 'Unix'],
        ext_modules=extensions,
        install_requires=build_requires,
        entry_points={
            'console_scripts': [
                # 'ModelHouse = aurochs.modelhouse.modelhousedb:_cli',
                # 'ResourceManager = aurochs.modelhouse.resource_manager:_cli',
                # 'ALS = aurochs.buffalo.buffalo:_cli_als',
                # 'W2V = aurochs.buffalo.buffalo:_cli_w2v',
                # 'ClusteringLDA = aurochs.farmer_john.farmer_john:_cli_lda',
                # 'AurochsApp = aurochs.app.aurochs_app:_cli'
            ]
        }
    )

    metadata['version'] = VERSION
    setup(**metadata)


if __name__ == '__main__':
    setup_package()
