'''
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
'''

import numpy as np
from mydtw import ddtw as DTW
# from mydtw import dtw as DTW


# Main function
# @param S: set of sequences
# @param n: no. of iterations
# @return: sample mean calculated with DBA
def dba_mean(S: np.ndarray, n: int) -> np.ndarray:

    # simple implementation: use first sequence as initial average
    # print(f'DBA Iteration 1 of {n}...')
    mean = perform_dba_iteration(np.copy(S[0]), S)

    for i in range(n-1):
        # print(f'DBA Iteration {i+2} of {n}...')
        mean = perform_dba_iteration(mean, S)

    # print('DBA Done.')
        
    return mean
        

# Calculate Associations Table
# @param c: initial average sequence
# @param S: set of sequences to be averaged
# @return array: each entry is a set of associate-coordinates from sequences in S
#	for each coordinate c_i of average
def calculate_associations(c: np.ndarray, S: np.ndarray) -> np.ndarray:

    A = np.zeros(c.size, dtype=object)	# associatons table

    # for each sequence s, get associations based on optimal warping path
    for i in range(S.shape[0]):
        s = S[i]
        cost, path = DTW(c, s)

        # iterate thru path, adding coordinate from s to corresponding list for c_i
        for j in range(path.shape[0]):
            a, b = path[j]	# indices for c & s at this point in path
            if A[a] == 0:
            	A[a] = {s[b]}
            else:
            	A[a] = A[a].union({s[b]})  # add s_b value to associations set for c_a

    return A


# For one iteration, calculate new average sequence based on assocations table
# @param c: sequence to be updated
# @param A: associations table of c
# @return average sequence
def calculate_average_update(c: np.ndarray, A: np.ndarray) -> np.ndarray:

    for i in range(c.size):
        c[i] = sum(A[i])/len(A[i])	# barycenter is just a fancy word for average

    return c


# Given a current average sequence & the set of all sequences, do one DBA iteration
def perform_dba_iteration(current_average: np.ndarray, sequences: np.ndarray) -> np.ndarray:

    associations_table = calculate_associations(current_average, sequences)
    updated_average_sequence = calculate_average_update(current_average, associations_table)
        
    return updated_average_sequence



