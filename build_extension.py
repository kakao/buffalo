import os
import platform
import sysconfig

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages

from cuda_extension import CUDA, build_ext

MAJOR = 1
MINOR = 2
MICRO = 2
Release = True
STAGE = {True: '', False: 'b'}.get(Release)
VERSION = f'{MAJOR}.{MINOR}.{MICRO}{STAGE}'
STATUS = {
    False: 'Development Status :: 4 - Beta',
    True: 'Development Status :: 5 - Production/Stable'
}

CLASSIFIERS = """{status}
Programming Language :: C++
Programming Language :: Cython
Programming Language :: Python
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
common_srcs = ['lib/algo.cc', "./3rd/json11/json11.cpp"]

if platform.system() == 'Darwin':
    print("to build for MacOS, gcc-11 version must be installed and set(for m1 support)")
    os.environ["CC"] = 'gcc-11'
    os.environ["CXX"] = 'g++-11'

def get_extend_compile_flags():
    flags = ['-march=native']
    return flags

extend_compile_flags = get_extend_compile_flags()


extensions = [
    Extension(name="buffalo.algo._als",
              sources=['buffalo/algo/_als.pyx', 'lib/algo_impl/als/als.cc'] + common_srcs,
              language='c++',
              include_dirs=['./include'] + EXTRA_INCLUDE_DIRS,
              libraries=['gomp'],
              library_dirs=LIBRARY_DIRS,
              runtime_library_dirs=LIBRARY_DIRS,
              extra_compile_args=['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + extend_compile_flags),
    Extension(name="buffalo.algo._cfr",
              sources=['buffalo/algo/_cfr.pyx', 'lib/algo_impl/cfr/cfr.cc'] + common_srcs,
              language='c++',
              include_dirs=['./include'] + EXTRA_INCLUDE_DIRS,
              libraries=['gomp'],
              library_dirs=LIBRARY_DIRS,
              runtime_library_dirs=LIBRARY_DIRS,
              extra_compile_args=['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + extend_compile_flags),
    Extension(name="buffalo.algo._bpr",
              sources=['buffalo/algo/_bpr.pyx', 'lib/algo_impl/bpr/bpr.cc'] + common_srcs,
              language='c++',
              include_dirs=['./include'] + EXTRA_INCLUDE_DIRS,
              libraries=['gomp'],
              library_dirs=LIBRARY_DIRS,
              runtime_library_dirs=LIBRARY_DIRS,
              extra_compile_args=['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + extend_compile_flags),
    Extension(name="buffalo.algo._plsi",
              sources=['buffalo/algo/_plsi.pyx', 'lib/algo_impl/plsi/plsi.cc'] + common_srcs,
              language='c++',
              include_dirs=['./include'] + EXTRA_INCLUDE_DIRS,
              libraries=['gomp'],
              library_dirs=LIBRARY_DIRS,
              runtime_library_dirs=LIBRARY_DIRS,
              extra_compile_args=['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + extend_compile_flags),
    Extension(name="buffalo.algo._warp",
              sources=['buffalo/algo/_warp.pyx', 'lib/algo_impl/warp/warp.cc'] + common_srcs,
              language='c++',
              include_dirs=['./include'] + EXTRA_INCLUDE_DIRS,
              libraries=['gomp'],
              library_dirs=LIBRARY_DIRS,
              runtime_library_dirs=LIBRARY_DIRS,
              extra_compile_args=['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + extend_compile_flags),
    Extension(name="buffalo.algo._w2v",
              sources=['buffalo/algo/_w2v.pyx', 'lib/algo_impl/w2v/w2v.cc'] + common_srcs,
              language='c++',
              include_dirs=['./include'] + EXTRA_INCLUDE_DIRS,
              libraries=['gomp'],
              library_dirs=LIBRARY_DIRS,
              runtime_library_dirs=LIBRARY_DIRS,
              extra_compile_args=['-fopenmp', '-std=c++14', '-ggdb', '-O3'] + extend_compile_flags),
    Extension(name="buffalo.misc._log",
              sources=["buffalo/misc/_log.pyx", "lib/misc/log.cc"] + common_srcs,
              language='c++',
              include_dirs=['./include'] + EXTRA_INCLUDE_DIRS,
              libraries=['gomp'],
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
              libraries=['gomp'],
              include_dirs=EXTRA_INCLUDE_DIRS,
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


class BuildExtension(build_ext):
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

        if platform.system() == 'Darwin':
            print("to build for MacOS, gcc-11 version must be installed and set(for m1 support)")
            os.environ["CC"] = 'gcc-11'
            os.environ["CXX"] = 'g++-11'
            cmake_args += ['-DCMAKE_C_COMPILER=gcc-11', '-DCMAKE_CXX_COMPILER=g++-11']
        build_args = []

        os.chdir(self.build_temp)
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.', '-j 4'] + build_args)
        os.chdir(str(cwd))


def build(kwargs):
    cmdclass = {'build_ext': build_ext}
    kwargs.update(
        dict(packages=find_packages(),
            cmdclass=cmdclass,
            ext_modules=cythonize(extensions),
            platforms=['Linux'],
            zip_safe=False,
        )
    )
