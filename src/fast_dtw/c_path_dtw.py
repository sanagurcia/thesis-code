import numpy.ctypeslib as npct
import ctypes as ct
import numpy as np

# Load library from shared object
lib_dtw = npct.load_library("libdtw.so", "/Users/santiago/thesis/code/src/fast_dtw")
# lib_dtw = npct.load_library("libdtw.so", "./src/fast_dtw") # also works, if cwd is thesis/code


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


def main():
    a = np.asarray([1, 1, 3, 10, 10], dtype="float32")
    b = np.asarray([1, 4, 4, 4, 8], dtype="float32")
    wp = dtw_path(a, b)
    print(type(a))
    print(type(b))
    print(wp)


if __name__ == "__main__":
    main()
