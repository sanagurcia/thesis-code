"""JITTED METHODS"""

import math
import numpy as np
from numba import njit
from .jit_dtw import dtw


def dba_mean(S: np.ndarray, n=3, verbose=False) -> np.ndarray:
    """Perform n iterations of DBA on set of sequences

    Args:
        S (np.ndarray): set of sequences
        n (int): no. of iterations
        verbose (bool): print info about iteration

    Returns:
        np.ndarray: sample mean calculated by DBA
    """

    if verbose:
        print(f"\nDBA Iteration 1 of {n}...")

    # use random sequence as initial average
    rand_i = np.random.randint(0, S.shape[0])
    mean = do_fast_dba_iteration(np.copy(S[rand_i]), S)

    for i in range(n - 1):
        if verbose:
            print(f"DBA Iteration {i+2} of {n}...")
        mean = do_fast_dba_iteration(mean, S)

    if verbose:
        print("DBA Done.")

    return mean


def do_fast_dba_iteration(seq_avg, S):
    B = np.zeros((seq_avg.size, seq_avg.size), dtype="float32")  # sparse MxM array

    for i in range(S.shape[0]):
        seq_s = S[i]
        _, path = dtw(seq_avg, seq_s)
        push_associations(B, seq_s, path)

    return update_average_jit(seq_avg, B)


@njit
def push_associations(A: np.ndarray, s: np.ndarray, path: np.ndarray):
    # A[avg_seq_index, [associations count, associatied-value1, associated-valued2... ]]

    # iterate thru path, adding coordinate-value from seq_s to corresponding list for seq_avg
    for j in range(path.shape[0]):
        avg_j, s_j = path[j]  # indices for seq_avg & seq_s at this point in path

        # first entry 'k' in row specifies current amount of associated-values in row
        A[avg_j, 0] = A[avg_j, 0] + 1  # increment k
        k = int(A[avg_j, 0])  # cast k as int
        A[avg_j, k] = s[s_j]  # assign associated value to index k


@njit
def update_average_jit(seq_avg: np.ndarray, A: np.ndarray) -> np.ndarray:
    # for each row i in A, sum up entries from A[i, 1:k+1], where k = A[i, 0]
    for i in range(seq_avg.size):
        row_a = A[i]
        k = int(row_a[0])
        seq_avg[i] = np.sum(row_a[1 : k + 1]) / k

    return seq_avg
