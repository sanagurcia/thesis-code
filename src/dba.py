"""
DBA Module
An implementation of the DTW Barycenter Algorithm, 
following pseudocode from Petitjean et al. (2011)

Each iteration consists of two steps:
1. Calculate associations table:
   For each sequences of S, use DTW to get matches for each coordinate of mean
2. Calculate average for each coordinate of mean:
   Take average of associated coordinates.

Hence, DTW serves to make explicit which coordinates of each sequence in S
are relevant for a particular coordinate in the mean.
Each coordinate of mean is merely the everyday average of all its aligned coordinates in S.

Functions:
    * dba_mean - perform DBA on set of sequences
"""

import numpy as np
from .fast_dtw import dtw_cost, dtw_path
from .shapedtw import shapedtw_cost, shapedtw_path

COL = "\033[92m"
CEND = "\033[0m"

# cost = dtw_cost
cost = shapedtw_cost
# path = dtw_path
path = shapedtw_path


def dba_mean(S: np.ndarray, iterations=3, verbose=False) -> np.ndarray:
    """Perform n iterations of DBA on set of sequences

    Args:
        S (np.ndarray): set of sequences
        n (int): no. of iterations
        verbose (bool): print info about iteration

    Returns:
        np.ndarray: sample mean calculated by DBA
    """

    # use random sequence as initial average
    rand_i = np.random.randint(0, S.shape[0])
    mean = np.copy(S[rand_i])
    if verbose:
        start_cost = calculate_average_cost_to_mean(mean, S)

    for i in range(iterations):
        if verbose:
            print(f"{COL}[DBA]{CEND} Iteration {i+1} of {iterations}...")
            avg_cost_before = calculate_average_cost_to_mean(mean, S)

        mean = perform_dba_iteration(mean, S)

        if verbose:
            avg_cost_after = calculate_average_cost_to_mean(mean, S)
            cost_change_percent = round(((avg_cost_before - avg_cost_after) / avg_cost_before * 100), 2)
            print(f"{COL}[DBA]{CEND} Cost reduction {cost_change_percent}%")

    if verbose:
        end_cost = calculate_average_cost_to_mean(mean, S)
        cost_change_percent = round(((start_cost - end_cost) / start_cost * 100), 2)
        print(f"{COL}[DBA]{CEND} Done. Total cost reduction {cost_change_percent}%")

    return mean


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

    A = np.zeros(seq_avg.size, dtype=object)  # associations table

    # for each sequence s, get associations based on optimal warping path
    for i in range(S.shape[0]):
        seq_s = S[i]
        wp = path(seq_avg, seq_s)

        # iterate thru path, adding coordinate from seq_s to corresponding list for seq_avg
        for j in range(wp.shape[0]):
            avg_j, s_j = wp[j]  # indices for seq_avg & seq_s at this point in path
            if A[avg_j] == 0:
                # if avg_j coordinate has no associations, init list with value at s_j
                A[avg_j] = [seq_s[s_j]]
            else:
                # add s_j value to associations list for avg_j
                A[avg_j].append(seq_s[s_j])

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
        assert A[i][0] != 0
        seq_avg[i] = sum(A[i]) / len(A[i])  # barycenter is just a fancy word for average

    return seq_avg


### UTILS


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
        total_cost += cost(current_mean, sequences[i])
    return total_cost / n_sequences
