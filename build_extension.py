import glob
import os
import platform
import re
import subprocess
import sys
from os.path import join as pjoin

import numpy as np
import packaging.version
from setuptools import Extension, find_packages

from cuda_extension import CUDA, build_ext

numpy_include_dirs = np.get_include()
EXTRA_INCLUDE_DIRS = [numpy_include_dirs,
                      "3rd/json11",
                      "3rd/spdlog/include",
                      "3rd/eigen3"]
common_srcs = ["lib/misc/log.cc", "lib/algo.cc", "./3rd/json11/json11.cpp"]

if platform.system().lower() == "darwin":

    def get_compiler(name: str):
        binaries = []
        if name == "gcc":
            pattern = re.compile(r"gcc-([0-9]+[.]*[0-9]*[.]*[0-9]*)")
        elif name == "g++":
            pattern = re.compile(r"g\+\+-([0-9]+[.]*[0-9]*[.]*[0-9]*)")
        for fname in glob.glob(pjoin(binary_dir, f"{name}*")):
            basename = os.path.basename(fname)
            if basename == name:
                ret = subprocess.run([fname, "--dumpversion"], capture_output=True)
                version = ret.stdout.strip().decode()
                binaries.append((fname, version))
            elif basename.startswith(f"{name}-"):
                matched = pattern.match(basename)
                if matched is None:
                    continue
                version = matched.group(1)
                binaries.append((fname, version))
        if not binaries:
            print("To build buffalo in MacOs, gcc must be installed. Install gcc via `brew install gcc`")
            sys.exit(1)

        binaries.sort(key=lambda x: packaging.version.Version(x[1]), reverse=True)
        return binaries[-1][0]

    ret = subprocess.run(["brew", "--prefix"], capture_output=True)
    if ret.stderr:
        print("`brew` is required. Install `brew` first.")
        sys.exit(1)
    brew_prefix = ret.stdout.strip().decode()
    binary_dir = pjoin(brew_prefix, "bin")
    # Find gcc & g++
    os.environ["CC"] = get_compiler("gcc")
    os.environ["CXX"] = get_compiler("g++")


def get_extend_compile_flags():
    flags = ["-march=native"]
    return flags


extend_compile_flags = get_extend_compile_flags()
extensions = [
    Extension(name="buffalo.algo._als",
              sources=["buffalo/algo/_als.pyx", "lib/algo_impl/als/als.cc"] + common_srcs,
              language="c++",
              include_dirs=["./include"] + EXTRA_INCLUDE_DIRS,
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-ggdb", "-O3"] + extend_compile_flags),
    Extension(name="buffalo.algo._cfr",
              sources=["buffalo/algo/_cfr.pyx", "lib/algo_impl/cfr/cfr.cc"] + common_srcs,
              language="c++",
              include_dirs=["./include"] + EXTRA_INCLUDE_DIRS,
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-ggdb", "-O3"] + extend_compile_flags),
    Extension(name="buffalo.algo._bpr",
              sources=["buffalo/algo/_bpr.pyx", "lib/algo_impl/bpr/bpr.cc"] + common_srcs,
              language="c++",
              include_dirs=["./include"] + EXTRA_INCLUDE_DIRS,
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-ggdb", "-O3"] + extend_compile_flags),
    Extension(name="buffalo.algo._plsi",
              sources=["buffalo/algo/_plsi.pyx", "lib/algo_impl/plsi/plsi.cc"] + common_srcs,
              language="c++",
              include_dirs=["./include"] + EXTRA_INCLUDE_DIRS,
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-ggdb", "-O3"] + extend_compile_flags),
    Extension(name="buffalo.algo._warp",
              sources=["buffalo/algo/_warp.pyx", "lib/algo_impl/warp/warp.cc"] + common_srcs,
              language="c++",
              include_dirs=["./include"] + EXTRA_INCLUDE_DIRS,
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-ggdb", "-O3"] + extend_compile_flags),
    Extension(name="buffalo.algo._w2v",
              sources=["buffalo/algo/_w2v.pyx", "lib/algo_impl/w2v/w2v.cc"] + common_srcs,
              language="c++",
              include_dirs=["./include"] + EXTRA_INCLUDE_DIRS,
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-ggdb", "-O3"] + extend_compile_flags),
    Extension(name="buffalo.misc._log",
              sources=["buffalo/misc/_log.pyx"] + common_srcs,
              language="c++",
              include_dirs=["./include"] + EXTRA_INCLUDE_DIRS,
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-ggdb", "-O3"] + extend_compile_flags),
    Extension(name="buffalo.data.fileio",
              sources=["buffalo/data/fileio.pyx"],
              language="c++",
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-ggdb", "-O3"] + extend_compile_flags),
    Extension(name="buffalo.parallel._core",
              sources=["buffalo/parallel/_core.pyx"],
              language="c++",
              libraries=["gomp"],
              include_dirs=EXTRA_INCLUDE_DIRS,
              extra_compile_args=["-fopenmp", "-std=c++14", "-ggdb", "-O3"] + extend_compile_flags),
]

if CUDA:
    extra_compile_args = ["-std=c++14", "-ggdb", "-O3"] + extend_compile_flags
    extensions.append(Extension("buffalo.algo.cuda._als",
                                sources=["buffalo/algo/cuda/_als.pyx",
                                         "lib/cuda/als/als.cu",
                                         "./3rd/json11/json11.cpp",
                                         "lib/misc/log.cc"],
                                language="c++",
                                extra_compile_args=extra_compile_args,
                                library_dirs=[CUDA["lib64"]],
                                libraries=["cudart", "cublas", "curand"],
                                include_dirs=["./include", numpy_include_dirs,
                                              CUDA["include"], "./3rd/json11",
                                              "./3rd/spdlog/include"]))
    extensions.append(Extension("buffalo.algo.cuda._bpr",
                                sources=["buffalo/algo/cuda/_bpr.pyx",
                                         "lib/cuda/bpr/bpr.cu",
                                         "./3rd/json11/json11.cpp",
                                         "lib/misc/log.cc"],
                                language="c++",
                                extra_compile_args=extra_compile_args,
                                library_dirs=[CUDA["lib64"]],
                                libraries=["cudart", "cublas", "curand"],
                                include_dirs=["./include", numpy_include_dirs,
                                              CUDA["include"], "./3rd/json11",
                                              "./3rd/spdlog/include"]))
else:
    print("Failed to find CUDA toolkit. Building without GPU acceleration.")


def build(kwargs):
    cmdclass = {"build_ext": build_ext}
    kwargs.update(
        dict(packages=find_packages(),
             cmdclass=cmdclass,
             ext_modules=extensions,
             platforms=["Linux", "MacOS"],
             zip_safe=False)
    )
