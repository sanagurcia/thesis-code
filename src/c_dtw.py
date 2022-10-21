"""
ctype wrapper for CLang-implemented DTW.
Expects libdtw.so (source object of dtw.c) in same directory.

Functions:
    * dtw_cost - calculate DTW distance for two sequences
"""

import ctypes
import numpy as np

# create CDLL instance of source object
_libdtw = ctypes.CDLL("src/libdtw.so")

# Specify argument types: (int, int, float*, float*); return type: float
_libdtw.dtw_cost.argtypes = (
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
)
_libdtw.dtw_cost.restype = ctypes.c_float


def dtw_cost(seq_a: np.ndarray, seq_b: np.ndarray) -> np.float32:
    """Wrapper for C-implemented dtw_cost()

    Args:
        seq_a (np.ndarray): first sequence
        seq_b (np.ndarray): second sequence

    Returns:
        np.float32: DTW cost
    """

    global _libdtw  # define as global "to help interpreter find it"
    a_length = seq_a.size
    b_length = seq_b.size

    # define ctypes array type: 'multiply' int-ctype with length
    # then, to create array with this type, call array_type(*seq)
    array_type_a = ctypes.c_float * a_length
    array_type_b = ctypes.c_float * b_length

    # call c func with c-typed arguments
    result = _libdtw.dtw_cost(
        ctypes.c_int(a_length),
        ctypes.c_int(b_length),
        array_type_a(*seq_a),
        array_type_b(*seq_b),
    )
    return np.float32(result)  # cast ctypes.c_float result to numpy float32
