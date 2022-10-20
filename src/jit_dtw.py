"""
JIT DTW Module

This module offers a speed up 1-2 orders of magnitude over non jitted DTW module!

Functions:
    * dtw - compute DTW on two sequences, returning cost & warping path (30x speedup)
    * dtw_cost - only compute and return cost (300x speedup)
"""

import numpy as np
from numba import njit, jit


def dtw(a: np.ndarray, b: np.ndarray) -> (float, np.ndarray):
    """Compute JIT DTW: return cost & warping path

    Args:
        a (ndarray): First sequence
        b (ndarray): Second sequence

    Returns:
        (float, 1D array): min cost, warping path
    """

    D, P = compute_dtw_array(a, b)  # compute DTW arrays
    wp = get_warping_path(P)
    return D[a.size - 1, b.size - 1], wp  # return total cost, warping path


# 20-30x speed up over non-jitted method!
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


@njit  # 10-20x speedup over non-jitted method
def get_warping_path(P: np.ndarray) -> (np.ndarray):
    """Extract warping path from Path array

    Args:
        P (np.ndarray): NxNx2 array from compute_dtw_array

    Returns:
        np.ndarray: Nx2 array
    """
    # P is a seq_a x seq_b array, with 2-length-tuples as entries

    # longest possible path is P length + width
    wp = np.zeros((P.shape[0] + P.shape[1], 2), dtype="uint16")

    # indices for end-cell in P
    i = P.shape[0] - 1
    j = P.shape[1] - 1

    # init warping path with last indices
    wp[0] = [i, j]

    k = 1
    while not (i == 0 and j == 0):
        wp[k] = P[i, j]  # insert predecessor into warping path
        i = wp[k, 0]  # move indices i, j based on
        j = wp[k, 1]
        k += 1

    return wp[:k]


@njit  # 10x speed up over jitted-dtw with warping path
def dtw_cost(a: np.ndarray, b: np.ndarray) -> float:
    """Get DTW distance between two sequences

    Args:
        a (np.ndarray): first sequence
        b (np.ndarray): second sequence

    Returns:
        float: total distance
    """

    D = np.zeros((a.size, b.size), dtype="float32")

    # init first cell
    D[0, 0] = abs(a[0] - b[0])

    # calculate dtw for first column
    for i in range(1, a.size):
        D[i, 0] = abs(a[i] - b[0]) + D[i - 1, 0]  # dtw-cost := current-cost + predecessor-cost

    # calculate dtw for first row
    for j in range(1, b.size):
        D[0, j] = abs(a[0] - b[j]) + D[0, j - 1]  # dtw-cost := current-cost + predecessor-cost

    # for all other entries
    for i in range(1, a.size):
        for j in range(1, b.size):
            d0 = D[i - 1, j]  # top
            d1 = D[i, j - 1]  # left
            d2 = D[i - 1, j - 1]  # diagonal

            # Find min predecessor
            d_min = d0
            if d1 < d0:
                if d1 < d2:
                    d_min = d1
                else:
                    d_min = d2
            elif d2 < d0:
                d_min = d2

            D[i, j] = abs(a[i] - b[j]) + d_min  # store cost

    return D[a.size - 1, b.size - 1]
