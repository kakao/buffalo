import glob
import logging
import os
import platform
import re
import subprocess
import sys
from os.path import join as pjoin

import cpuinfo
import numpy as np
import packaging.version
from setuptools import Extension, setup

sys.path.append(os.path.join(os.path.dirname(__file__), "install"))
from cuda_setup import CUDA, build_ext

numpy_include_dirs = np.get_include()
extra_include_dirs = [numpy_include_dirs, "3rd/json11", "3rd/spdlog/include", "3rd/eigen3"]
common_srcs = ["lib/misc/log.cc", "lib/algo.cc", "./3rd/json11/json11.cpp"]

# NOTE: buffalo needs gcc/g++ for compilation since it uses gnu's parallel sort implementation.
# Clang does not support parallel sort so far.
# C++17's parallel algorithm needs TBB library as threading backend which is only available for Intel CPU.
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
            logging.error("To build buffalo in MacOs, gcc must be installed. Install gcc via `brew install gcc`")
            sys.exit(1)

        binaries.sort(key=lambda x: packaging.version.Version(x[1]), reverse=True)
        return binaries[-1][0]

    ret = subprocess.run(["brew", "--prefix"], capture_output=True)
    if ret.stderr:
        logging.error("`brew` is required to check and install gcc/g++. Install `brew` first.")
        sys.exit(1)
    brew_prefix = ret.stdout.strip().decode()
    binary_dir = pjoin(brew_prefix, "bin")
    # Find gcc & g++
    os.environ["CC"] = get_compiler("gcc")
    os.environ["CXX"] = get_compiler("g++")


# NOTE: In Intel's CPU, Eigen enables SSE2 on x86_64 arch by default.
# But most of modern CPU also supports AVX2 FMA which boosts performance.
# Thus enable these flags if possible.
# In ARM NEOM, Eigen enables SIMD by default. Thus no need to manually set flags.
cpu_info = cpuinfo.get_cpu_info()
arch = cpu_info.get("arch", "").lower()
extended_compile_flags = []
if arch == "x86_64":
    flags = cpu_info["flags"]
    if "avx2" in flags:
        extended_compile_flags.append("-mavx2")
    if "fma" in flags:
        extended_compile_flags.append("-mfma")


extensions = [
    Extension(name="buffalo.algo._als",
              sources=["buffalo/algo/_als.pyx", "lib/algo_impl/als/als.cc"] + common_srcs,
              language="c++",
              include_dirs=["./include"] + extra_include_dirs,
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-O3"] + extended_compile_flags,
              define_macros=[("NPY_NO_DEPRECATED_API", "1")]),
    Extension(name="buffalo.algo._cfr",
              sources=["buffalo/algo/_cfr.pyx", "lib/algo_impl/cfr/cfr.cc"] + common_srcs,
              language="c++",
              include_dirs=["./include"] + extra_include_dirs,
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-O3"] + extended_compile_flags,
              define_macros=[("NPY_NO_DEPRECATED_API", "1")]),
    Extension(name="buffalo.algo._bpr",
              sources=["buffalo/algo/_bpr.pyx", "lib/algo_impl/bpr/bpr.cc"] + common_srcs,
              language="c++",
              include_dirs=["./include"] + extra_include_dirs,
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-O3"] + extended_compile_flags,
              define_macros=[("NPY_NO_DEPRECATED_API", "1")]),
    Extension(name="buffalo.algo._plsi",
              sources=["buffalo/algo/_plsi.pyx", "lib/algo_impl/plsi/plsi.cc"] + common_srcs,
              language="c++",
              include_dirs=["./include"] + extra_include_dirs,
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-O3"] + extended_compile_flags,
              define_macros=[("NPY_NO_DEPRECATED_API", "1")]),
    Extension(name="buffalo.algo._warp",
              sources=["buffalo/algo/_warp.pyx", "lib/algo_impl/warp/warp.cc"] + common_srcs,
              language="c++",
              include_dirs=["./include"] + extra_include_dirs,
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-O3"] + extended_compile_flags,
              define_macros=[("NPY_NO_DEPRECATED_API", "1")]),
    Extension(name="buffalo.algo._w2v",
              sources=["buffalo/algo/_w2v.pyx", "lib/algo_impl/w2v/w2v.cc"] + common_srcs,
              language="c++",
              include_dirs=["./include"] + extra_include_dirs,
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-O3"] + extended_compile_flags,
              define_macros=[("NPY_NO_DEPRECATED_API", "1")]),
    Extension(name="buffalo.misc._log",
              sources=["buffalo/misc/_log.pyx"] + common_srcs,
              language="c++",
              include_dirs=["./include"] + extra_include_dirs,
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-O3"] + extended_compile_flags),
    Extension(name="buffalo.data.fileio",
              sources=["buffalo/data/fileio.pyx"],
              language="c++",
              libraries=["gomp"],
              extra_compile_args=["-fopenmp", "-std=c++14", "-O3"] + extended_compile_flags),
    Extension(name="buffalo.parallel._core",
              sources=["buffalo/parallel/_core.pyx"],
              language="c++",
              libraries=["gomp"],
              include_dirs=extra_include_dirs,
              extra_compile_args=["-fopenmp", "-std=c++14", "-O3"] + extended_compile_flags,
              define_macros=[("NPY_NO_DEPRECATED_API", "1")]),
]

if CUDA:
    extra_compile_args = ["-std=c++14", "-O3"] + extended_compile_flags
    extensions.append(
        Extension(
            name="buffalo.algo.cuda._als",
            sources=[
                "buffalo/algo/cuda/_als.pyx",
                "lib/cuda/als/als.cu",
                "./3rd/json11/json11.cpp",
                "lib/misc/log.cc"
            ],
            language="c++",
            extra_compile_args=extra_compile_args,
            library_dirs=[CUDA["lib64"]],
            libraries=["cudart", "cublas", "curand"],
            include_dirs=[
                "./include", numpy_include_dirs, CUDA["include"], "./3rd/json11", "./3rd/spdlog/include"
            ],
            define_macros=[("NPY_NO_DEPRECATED_API", "1")],
        )
    )
    extensions.append(
        Extension(
            name="buffalo.algo.cuda._bpr",
            sources=[
                "buffalo/algo/cuda/_bpr.pyx",
                "lib/cuda/bpr/bpr.cu",
                "./3rd/json11/json11.cpp",
                "lib/misc/log.cc"
            ],
            language="c++",
            extra_compile_args=extra_compile_args,
            library_dirs=[CUDA["lib64"]],
            libraries=["cudart", "cublas", "curand"],
            include_dirs=[
                "./include", numpy_include_dirs, CUDA["include"], "./3rd/json11", "./3rd/spdlog/include"
            ],
            define_macros=[("NPY_NO_DEPRECATED_API", "1")],
        )
    )
else:
    logging.info("Failed to find CUDA toolkit. Building without GPU acceleration.")


setup(cmdclass={"build_ext": build_ext}, ext_modules=extensions)
