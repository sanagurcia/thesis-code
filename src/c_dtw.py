"""
ctype wrapper for CLang-implemented DTW.
Expects libdtw.so (source object of dtw.c) in same directory.

Functions:
    * dtw_cost - calculate DTW distance for two sequences
"""

import ctypes
import numpy as np

# import numpy.ctypeslib as npct


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

    a_length = seq_a.size
    b_length = seq_b.size

    # define ctypes array type: 'multiply' int-ctype with length
    # then, to create array with this type, call array_type(*seq)
    array_type_a = ctypes.c_float * a_length
    array_type_b = ctypes.c_float * b_length

    # call c func with c-typed arguments
    cost = _libdtw.dtw_cost(
        ctypes.c_int(a_length),
        ctypes.c_int(b_length),
        array_type_a(*seq_a),
        array_type_b(*seq_b),
    )
    return np.float32(cost)  # cast ctypes.c_float result to numpy float32


# Different approach for dtw_path
# lib_dtw = npct.load_library("libdtw.so", "/Users/santiago/thesis/code/src")


# def dtw_path(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
#     # longest possible warping path is a_len + b_len
#     # allocate warping path
#     wp = np.zeros((seq_a.size + seq_b.size) * 2, dtype="uint16")

#     # Define ct types
#     c_floatp = ctypes.POINTER(ctypes.c_float)  # ctype float*
#     c_uintp = ctypes.POINTER(ctypes.c_uint16)  # ctype uint16*

#     lib_dtw.dtw_path(
#         seq_a.size,  # python int not need to be cast
#         seq_b.size,
#         seq_a.ctypes.data_as(c_floatp),  # cast np arrays to ctype float pointers
#         seq_b.ctypes.data_as(c_floatp),
#         wp.ctypes.data_as(c_uintp),  # index wp with step=2
#     )

#     wp = np.append(wp[wp != 0], [0, 0])  # remove all zeros except last pair
#     return wp.reshape((int(wp.size / 2), 2))  # reshape warping path to nx2 array
