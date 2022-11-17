import ctypes as ct
import numpy.ctypeslib as npct
import numpy as np

# Load library from shared object
lib_dtw = npct.load_library("libdtw.so", "./src/fast_dtw")  # cwd must be thesis/code
lib_multi_dtw = npct.load_library("libmultidtw.so", "./src/fast_dtw")  # cwd must be thesis/code


def dtw_path(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    # Define ctypes
    c_floatp = ct.POINTER(ct.c_float)  # float*
    c_uintp = ct.POINTER(ct.c_uint16)  # unsigned short*

    # allocate warping path as numpy array
    wp = np.zeros((seq_a.size + seq_b.size) * 2, dtype="uint16")

    # call c func with casted c-typed arguments, using ndarray.ctypes.data_as()
    wp_length = lib_dtw.dtw_path(
        seq_a.size,  # a_len
        seq_b.size,  # b_len
        seq_a.ctypes.data_as(c_floatp),  # *a
        seq_b.ctypes.data_as(c_floatp),  # *b
        wp.ctypes.data_as(c_uintp),  # *wp
    )
    wp_shape = (int(wp_length / 2), 2)
    return wp[:wp_length].reshape(wp_shape)


def dtw_cost(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    # Define ctypes
    c_floatp = ct.POINTER(ct.c_float)  # float*
    lib_dtw.dtw_cost.restype = ct.c_float  # define return type float

    # call c func with casted c-typed arguments, using ndarray.ctypes.data_as()
    cost = lib_dtw.dtw_cost(
        seq_a.size,  # a_len
        seq_b.size,  # b_len
        seq_a.ctypes.data_as(c_floatp),  # *a
        seq_b.ctypes.data_as(c_floatp),  # *b
    )
    return cost


def multi_dtw_cost(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Multidimensional dtw cost"""

    # Assert that sequences are 2-D and that the second dimensions are equal
    assert len(seq_a.shape) == len(seq_b.shape) == 2
    assert seq_a.shape[1] == seq_b.shape[1]

    # Define ctypes
    c_floatp = ct.POINTER(ct.c_float)  # float*
    lib_multi_dtw.dtw_cost.restype = ct.c_float  # define return type float

    # call c func with casted c-typed arguments, using ndarray.ctypes.data_as()
    cost = lib_multi_dtw.dtw_cost(
        seq_a.shape[0],  # a_len
        seq_b.shape[0],  # b_len
        seq_a.shape[1],  # dimension
        seq_a.ctypes.data_as(c_floatp),  # *a
        seq_b.ctypes.data_as(c_floatp),  # *b
    )
    return cost


def multi_dtw_path(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    """Multidimensional dtw path"""

    # Assert that sequences are 2-D and that the second dimensions are equal
    assert len(seq_a.shape) == len(seq_b.shape) == 2
    assert seq_a.shape[1] == seq_b.shape[1]

    seq_a_len = seq_a.shape[0]
    seq_b_len = seq_b.shape[0]

    # Define ctypes
    c_floatp = ct.POINTER(ct.c_float)  # float*
    c_uintp = ct.POINTER(ct.c_uint16)  # unsigned short*

    # allocate warping path as numpy array
    wp = np.zeros((seq_a_len + seq_b_len) * 2, dtype="uint16")

    # call c func with casted c-typed arguments, using ndarray.ctypes.data_as()
    wp_length = lib_multi_dtw.dtw_path(
        seq_a_len,  # a_len
        seq_b_len,  # b_len
        seq_a.shape[1],  # 2 dimension length
        seq_a.ctypes.data_as(c_floatp),  # *a
        seq_b.ctypes.data_as(c_floatp),  # *b
        wp.ctypes.data_as(c_uintp),  # *wp
    )
    wp_shape = (int(wp_length / 2), 2)
    return wp[:wp_length].reshape(wp_shape)


def main():
    a = np.arange(12, dtype="float32").reshape((4, 3))
    b = np.arange(1, 13, dtype="float32").reshape((4, 3))
    wp = multi_dtw_path(a, b)
    print(wp)


if __name__ == "__main__":
    main()
