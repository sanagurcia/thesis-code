"""DTW Module

Module for computing dynamic time warping distance between two sequences.
Follows pseudo code from Petitjean et al (2011).

Useful functions:

    * dtw: computes vanilla DTW on two sequences
    * ddtw: computes derivative-DTW on two sequences
"""

import numpy as np


def dtw(a: np.ndarray, b: np.ndarray) -> (float, np.ndarray):
    """Compute vanilla DTW

    Args:
        a (ndarray): First sequence
        b (ndarray): Second sequence

    Returns:
        (float, 1D array): min cost, warping path
    """

    D = compute_dtw_array(a, b, euclidean_delta)
    return dtw_cost_path(D)


def ddtw(a: np.ndarray, b: np.ndarray) -> (float, np.ndarray):
    """Compute DTW using estimated derivatives as distance-measure

    Method originally proposed by Keogh et al. (2001)

    Args:
        a (ndarray): First sequence
        b (ndarray): Second sequence

    Returns:
        (float, 1D array): min cost, warping path
    """

    dD = compute_dtw_array(a, b, derivative_delta)
    return dtw_cost_path(dD)


def dtw_cost_path(D: np.ndarray) -> (float, np.ndarray):
    """Extract total cost and warping path from DTW array

    Args:
        D (np.ndarray): Computed DTW array

    Returns:
        (float, 1D-array): total cost, list of warping path indices
    """

    m = D.shape[0]
    n = D.shape[1]

    # get total cost
    dtw_distance = D[m - 1, n - 1][0]

    # get warping path
    i = [m - 1, n - 1]  # end-index on path (i, j)
    warping_path = [i]  # initialize path with end-index

    # construct warping-path by traversing DTW array backwards
    while not (i[0] == 0 and i[1] == 0):
        j = i[0]
        k = i[1]
        i = D[j][k][1]  # move index to predecessor
        warping_path.insert(0, i)  # push new index into stack

    warping_path = np.asarray(warping_path)  # convert list to np.array

    return (dtw_distance, warping_path)


def compute_dtw_array(a: np.ndarray, b: np.ndarray, delta) -> np.ndarray:
    """Populate DTW array for two sequences & given distance-measure (delta)

    Parametrizing delta enables use for both vanilla DTW & DDTW

    Args:
        a (np.ndarray): first sequence
        b (np.ndarray): second sequence
        delta (function): distance measure

    Returns:
        np.ndarray: 2D ndarray
        Entry contains accumulated cost & predecessor-index on warping path [cost, (i,j)]
    """

    # init target array: each entry consists of [float, (int, int)]
    D = np.zeros((a.size, b.size), dtype="float32, 2int32")

    # init first entry
    d = delta(a, 0, b, 0)
    D[0, 0] = (d, (0, 0))

    # calculate dtw for first column
    for i in range(1, a.size):
        pre = (i - 1, 0)
        d = delta(a, i, b, 0) + D[pre][0]  # dtw is predecessor-cost + current-cost
        D[i, 0] = (d, pre)  # store 'pre' as predecessor index for path

    # calculate dtw for first row
    for j in range(1, b.size):
        pre = (0, j - 1)
        d = delta(a, 0, b, j) + D[pre][0]
        D[0, j] = (d, pre)

    # for all other entries (notice double for loop!)
    for i in range(1, a.size):
        for j in range(1, b.size):
            (pre_cost, pre_index) = special_min(i, j, D)
            d = delta(a, i, b, j) + pre_cost
            D[i, j] = (d, pre_index)
    return D


def euclidean_delta(a: np.ndarray, a_i: int, b: np.ndarray, b_i: int) -> float:
    """Calculate vanilla distance (float) between single entry from pair of sequencies

    Args:
        a (np.ndarray): First sequence
        a_i (int): Index of first sequence
        b (np.ndarray): Second sequence
        b_i (int): Index of second sequence

    Returns:
        float: Distance
    """
    return abs(a[a_i] - b[b_i])


def derivative_delta(a: np.ndarray, a_i: int, b: np.ndarray, b_i: int) -> float:
    """Calculate derivative distance (float) between single entry from pair of sequencies

    For each point to be compared,
    estimates derivative around point and returns difference between derivatives.
    Following Keogh et al. (2001)

    Args:
        a (np.ndarray): First sequence
        a_i (int): Index of first sequence
        b (np.ndarray): Second sequence
        b_i (int): Index of second sequence

    Returns:
        float: Distance
    """

    # if boundary cases, use estimate of next/previous index
    if a_i == 0:
        a_i = 1
    elif a_i == a.shape[0] - 1:
        a_i = a.shape[0] - 2
    if b_i == 0:
        b_i = 1
    elif b_i == b.shape[0] - 1:
        b_i = b.shape[0] - 2

    # estimate derivatives at a[a_i] & b[b_i],
    #   where estimate := average between two slopes:
    #   1) index & left-neighbor, 2) right-neighbor & left-neighbor
    d_a = (a[a_i] - a[a_i - 1] + ((a[a_i + 1] - a[a_i - 1]) / 2)) / 2
    d_b = (b[b_i] - b[b_i - 1] + ((b[b_i + 1] - b[b_i - 1]) / 2)) / 2

    # difference of estimated derivatives
    return abs(d_a - d_b)


def special_min(i: int, j: int, D: np.ndarray) -> [float, (int, int)]:
    """Find minimum cost predecessor for given index-tuple (i, j)

    Args:
        i, j (int): current index
        D (ndarray): currently computed DTW array

    Returns:
        [float, (int, int)]: cost, predecessor index-tuple
    """

    x = [D[i - 1, j - 1][0], (i - 1, j - 1)]  # object: [cost, (i, j)]
    y = [D[i, j - 1][0], (i, j - 1)]
    z = [D[i - 1, j][0], (i - 1, j)]

    A = np.array([x, y, z], dtype=object)
    m = np.argmin(A[:, 0])  # find entry in A with minimum cost

    return A[m]

