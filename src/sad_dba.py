"""
SAD's implementation of the DTW Barycenter Algorithm
Following pseudocode from Petitjean et al. (2011)

Each iteration consists of two steps:
1. Calculate associations table:
   For each sequences of S, use DTW to get matches for each coordinate of mean
2. Calculate average for each coordinate of mean:
   Take average of associated coordinates.

Hence, DTW serves to make explicit which coordinates of each sequence in S
are relevant for a particular coordinate in the mean.
Each coordinate of mean is merely the everyday average of all its aligned coordinates in S.
"""

import numpy as np
from src.sad_dtw import dtw


def dba_mean(S: np.ndarray, n: int, verbose=False) -> np.ndarray:
    """Perform n iterations of DBA on set of sequences

    Args:
        S (np.ndarray): set of sequences
        n (int): no. of iterations
        verbose (bool): print info about iteration

    Returns:
        np.ndarray: sample mean calculated by DBA
    """

    if verbose:
        print(f"DBA Iteration 1 of {n}...")

    # simple implementation: use first sequence as initial average
    mean = perform_dba_iteration(np.copy(S[0]), S)

    for i in range(n - 1):
        if verbose:
            print(f"DBA Iteration {i+2} of {n}...")
        mean = perform_dba_iteration(mean, S)

    if verbose:
        print("DBA Done.")

    return mean


def calculate_associations(seq_avg: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Calculate associations table

    For each sequence seq_s in S:
        - use DTW to align average sequence with seq_s
        - for each entry j from alignment/warping-path:
            - include s_j coordinate in associations lsit of avg_j coordinate

    Thus each coordinate in seq_avg is associated with a set of aligned coordinates from all seq_s in S.

    Runtime complexity: O(n*m^2)
        n := no. of sequences to average
        m := length of each sequence
    complexity =~ n [outer-loop] * (m^2 [dtw] + m [inner-loop]) =~ n * (m^2 + m) => O(n*m^2)

    Args:
        seq_avg (np.ndarray): current average sequence
        S (np.ndarray): set of sequences to be averaged

    Returns:
        np.ndarray: associations table
    """

    A = np.zeros(seq_avg.size, dtype=object)  # associatons table

    # for each sequence s, get associations based on optimal warping path
    for i in range(S.shape[0]):
        seq_s = S[i]
        _, path = dtw(seq_avg, seq_s)

        # iterate thru path, adding coordinate from seq_s to corresponding list for seq_avg
        for j in range(path.shape[0]):
            avg_j, s_j = path[j]  # indices for seq_avg & seq_s at this point in path
            if A[avg_j] == 0:
                A[avg_j] = {
                    seq_s[s_j]
                }  # if avg_j coordinate has no associations, init set with value at s_j
            else:
                A[avg_j] = A[avg_j].union(
                    {seq_s[s_j]}
                )  # add s_j value to associations set for avg_j

    return A


def calculate_average_update(seq_avg: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Calculate new average sequence based on associations table

    Args:
        seq_avg (np.ndarray): current average sequence
        A (np.ndarray): associations table

    Returns:
        np.ndarray: new average sequence
    """

    for i in range(seq_avg.size):
        seq_avg[i] = sum(A[i]) / len(A[i])  # barycenter is just a fancy word for average

    return seq_avg


def perform_dba_iteration(current_average: np.ndarray, sequences: np.ndarray) -> np.ndarray:
    """Given current avg sequence & set of all sequences, do one DBA iteration
    Args:
        current_average (np.ndarray): average sequence
        sequences (np.ndarray): all sequences

    Returns:
        np.ndarray: updated average sequence
    """

    associations_table = calculate_associations(current_average, sequences)
    updated_average_sequence = calculate_average_update(current_average, associations_table)

    return updated_average_sequence


def calculate_average_cost_to_mean(current_mean: np.ndarray, sequences: np.ndarray) -> float:
    """(Sanity check purposes) Return average cost from mean to set of sequences

    Args:
        current_mean (np.ndarray)
        sequences (np.ndarray)

    Returns:
        float: average cost
    """
    total_cost = 0
    n_sequences = sequences.shape[0]
    for i in range(n_sequences):
        cost, _ = dtw(current_mean, sequences[i])
        total_cost += cost
    return total_cost / n_sequences
