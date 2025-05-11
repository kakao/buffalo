import ctypes
import ctypes.util
from typing import Optional

BLAS_LIB_NAMES = [
    "openblas",  # OpenBLAS
    "mkl_rt",  # Intel MKL (runtime)
    "blas",  # Generic
]


def _locate_blas() -> Optional[str]:
    """
    Attempts to find a BLAS library.

    This function iterates over a list of potential BLAS library names (e.g., 'openblas',
    'mkl_rt', etc.). It first checks if the library can be loaded, then tries to find
    a valid function within the library.

    Returns:
        str: The name of the BLAS library that was successfully loaded, or None if no
             library could be found.
    """

    loaded_lib = None
    found_lib = None

    for lib_name in BLAS_LIB_NAMES:
        path = ctypes.util.find_library(lib_name)
        if path:
            try:
                # On Linux, find_library often returns just the name (e.g., 'libblas.so.3')
                # On macOS, it might be a full path.
                loaded_lib = ctypes.CDLL(path)
                found_lib = lib_name
                break
            except OSError:
                pass

    if loaded_lib:
        try:
            ssyrk_func = getattr(loaded_lib, "ssyrk_")  # Note the trailing underscore
        except AttributeError:
            try:
                ssyrk_func = getattr(loaded_lib, "cblas_ssyrk")  # cblas interface
            except AttributeError:
                pass
    return found_lib


blas_lib_name = _locate_blas()
