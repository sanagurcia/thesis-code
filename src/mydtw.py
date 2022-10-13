# Script for calculating dynamic time warping distance between two sequences
# Following pseudo code from Petitjean et al (2011)
# Santiago Agurcia -- 04.10.2021

import numpy as np


# Calculate regular DTW
def dtw(a: np.ndarray, b: np.ndarray) -> (float, [np.ndarray]):

    D = compute_dtw_array(a, b, euclidean_delta)
    return dtw_cost_path(D)


# Calculate DTW with estimated derivatives as distance metric
# According to method proposed by Keogh et al. (2001)
def ddtw(a: np.ndarray, b: np.ndarray) -> (float, [np.ndarray]):

    dD = compute_dtw_array(a, b, derivative_delta)
    return dtw_cost_path(dD)


# Given DTW array, extract total cost and warping path
# @return (total cost, Nx2-array of indices of warping path)
def dtw_cost_path(D: np.ndarray) -> (float, [np.ndarray]):

    m = D.shape[0]
    n = D.shape[1]

    # get cost
    dtw_distance = D[m - 1, n - 1][0]

    # get warping path
    i = [m - 1, n - 1]
    warping_path = [i]  # initialize path with last index

    while not (i[0] == 0 and i[1] == 0):
        j = i[0]
        k = i[1]
        i = D[j][k][1]  # move index to predecessor
        warping_path.insert(0, i)  # push new index into stack

    warping_path = np.asarray(warping_path)  # convert list to np.array

    return (dtw_distance, warping_path)


# @param a: first sequence
# @param b: second sequence
# @return: array[cost, (path_index)]
def compute_dtw_array(a: np.ndarray, b: np.ndarray, delta) -> np.ndarray:

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


# Distance functions take the sequences and particular indices as arguments
def euclidean_delta(a: np.ndarray, a_i: int, b: np.ndarray, b_i: int) -> float:
    return abs(a[a_i] - b[b_i])


# estimate of derivatives distance
def derivative_delta(a: np.ndarray, a_i: int, b: np.ndarray, b_i: int) -> float:

    # if boundary cases, use estimate of next/previous index
    if a_i == 0:
        a_i = 1
    elif a_i == a.shape[0] - 1:
        a_i = a.shape[0] - 2
    if b_i == 0:
        b_i = 1
    elif b_i == b.shape[0] - 1:
        b_i = b.shape[0] - 2

    # estimate derivatives at a[a_i] & b[b_i]
    # average between two slopes:
    # 1) index & left-neighbor, 2) right-neighbor & left-neighbor
    d_a = (a[a_i] - a[a_i - 1] + ((a[a_i + 1] - a[a_i - 1]) / 2)) / 2
    d_b = (b[b_i] - b[b_i - 1] + ((b[b_i + 1] - b[b_i - 1]) / 2)) / 2

    # difference of estimated derivatives
    return abs(d_a - d_b)


# finds minimum cost predecessor
# returns (cost, index-tuple)
def special_min(i: int, j: int, D: np.ndarray) -> [float, (int, int)]:

    x = [D[i - 1, j - 1][0], (i - 1, j - 1)]
    y = [D[i, j - 1][0], (i, j - 1)]
    z = [D[i - 1, j][0], (i - 1, j)]

    A = np.array([x, y, z], dtype=object)
    m = np.argmin(A[:, 0])  # find index of entry with minimum cost

    return A[m]
