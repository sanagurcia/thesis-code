"""
JIT DTW Module
TODO:
    jitted method for only returning cost
    jitted method for extracting path
"""

import numpy as np
from numba import njit


def dtw(a: np.ndarray, b: np.ndarray) -> (float, np.ndarray):
    """Compute JIT DTW

    Args:
        a (ndarray): First sequence
        b (ndarray): Second sequence

    Returns:
        (float, 1D array): min cost, warping path
    """

    D, _ = compute_dtw_array(a, b)  # compute DTW arrays
    return D[a.size - 1, b.size - 1]  # return total cost


# 20-30x speed up!
@njit
def compute_dtw_array(a: np.ndarray, b: np.ndarray) -> (np.ndarray, np.ndarray):
    """Populate D(istance) and P(ath) arrays for two sequences, using euclidean distance"""

    # D(istance) is 2D array of type float: indexed by D[i, j] -> cost
    # P(ath) is 3D array of type int: indexed by P[i, j] -> previous indices [p_i, p_j] on path

    # init target array: each entry consists of [float, (int, int)]
    D = np.zeros((a.size, b.size), dtype="float32")
    P = np.zeros((a.size, b.size, 2), dtype="uint16")  # 3d-array, entries are [int, int]

    # init first cell
    D[0, 0] = abs(a[0] - b[0])
    P[0, 0] = [0, 0]

    # calculate dtw for first column
    for i in range(1, a.size):
        D[i, 0] = abs(a[i] - b[0]) + D[i - 1, 0]  # dtw-cost := current-cost + predecessor-cost
        P[i, 0] = [i - 1, 0]  # store predecessor indices

    # calculate dtw for first row
    for j in range(1, b.size):
        D[0, j] = abs(a[0] - b[j]) + D[0, j - 1]  # dtw-cost := current-cost + predecessor-cost
        P[0, j] = [0, j - 1]  # store predecessor indices

    # for all other entries
    for i in range(1, a.size):
        for j in range(1, b.size):
            # Store possible cell indices: top, left, diagonal

            d0 = D[i - 1, j]  # top
            d1 = D[i, j - 1]  # left
            d2 = D[i - 1, j - 1]  # diagonal

            # Find min predecessor
            pre = [i - 1, j]
            d_min = d0
            if d1 < d0:
                if d1 < d2:
                    pre = [i, j - 1]
                    d_min = d1
                else:
                    pre = [i - 1, j - 1]
                    d_min = d2
            elif d2 < d0:
                pre = [i - 1, j - 1]
                d_min = d2

            D[i, j] = abs(a[i] - b[j]) + d_min  # store cost
            P[i, j] = pre  # store predecessor

    return D, P


# def dtw_cost_path(D: np.ndarray) -> (float, np.ndarray):
#     """Extract total cost and warping path from DTW array

#     Args:
#         D (np.ndarray): Computed DTW array

#     Returns:
#         (float, 1D-array): total cost, list of warping path indices
#     """

#     m = D.shape[0]
#     n = D.shape[1]

#     # get total cost
#     dtw_distance = D[m - 1, n - 1][0]

#     # get warping path
#     i = [m - 1, n - 1]  # end-index on path (i, j)
#     warping_path = [i]  # initialize path with end-index

#     # construct warping-path by traversing DTW array backwards
#     while not (i[0] == 0 and i[1] == 0):
#         j = i[0]
#         k = i[1]
#         i = D[j][k][1]  # move index to predecessor
#         warping_path.insert(0, i)  # push new index into stack

#     warping_path = np.asarray(warping_path)  # convert list to np.array

#     return (dtw_distance, warping_path)
