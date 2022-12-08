"""
K-Means with DBA Module

This module implements the k-means clustering algorithm for application on time-series.
It calculates distance from datapoints to centroids based on the DTW distance,
and calculates centroids based on the DBA method.

Functions:
    * find_k_clusters - perform k-means on set of sequences
"""

import numpy as np
from .fast_dtw import dtw_cost
from .dba import DBA

DBA_ITERATIONS = 3
COL = "\033[96m"
CEND = "\033[0m"


def find_k_clusters(S: np.ndarray, k: int, verbose=False) -> ([[int]], np.ndarray):
    """Find k clusters in set of sequences

    Performs k-means algorithm:
        initialize k random centroids
        while centroids don't converge, perform iteration

    Args:
        S (np.ndarray): set of sequences
        k (int): no. of clusters

    Returns:
        ([[int]], ndarray): list of clusters, mean sequence for each cluster
    """

    # intialize k random centroids
    indices = np.random.randint(0, S.shape[0], k)
    C = np.zeros((k, S.shape[1]))
    for i in range(indices.size):
        C[i] = S[indices[i]]

    if verbose:
        print(f"\n{COL}[k-means]{CEND} Iteration {1}...")

    clusters, C_updated = k_means_iteration(C, S, verbose)  # perform first iteration

    # while centroids don't converge, perform iteration
    j = 1
    while not np.allclose(C, C_updated) and j < 5:
        if verbose:
            j += 1
            print(f"\n{COL}[k-means]{CEND} Iteration {j}...")

        C = np.copy(C_updated)
        clusters, C_updated = k_means_iteration(C, S, verbose)

    if verbose:
        print(f"\n{COL}[k-means]{CEND} Done.")

    return clusters, C_updated


def k_means_iteration(centroids: np.ndarray, S: np.ndarray, verbose=False) -> ([[int]], np.ndarray):
    """Perform k-means iteration

        for each sequence:
            compute distance (DTW) to each centroid
            assign sequence to closest centroid
        recompute centroids, using DBA

    Args:
        C (np.ndarray): candidate centroids
        S (np.ndarray): set of sequences

    Returns:
        (clusters, centroids): k lists of indices from S, array of mean sequences
    """

    k = centroids.shape[0]
    n_seq = S.shape[0]

    ##### CREATE CLUSTERS WITH DTW COST #####
    # create list of k lists for each cluster
    # each list will contain assigned sequence indices s_j of S
    clusters = [[] for i in range(k)]

    if verbose:
        print(f"{COL}[k-means]{CEND} Assigning sequences to nearest centroids...")

    # assign each sequence to cluster with closest centroid
    for j in range(n_seq):  # for each sequence,
        candidate_c = (-1, np.inf)  # candidate centroid: (index, cost)

        for i in range(k):  # for each centroid
            cost = dtw_cost(centroids[i], S[j])  # compute distance from seq to centroid

            if cost < candidate_c[1]:
                candidate_c = (i, cost)

        clusters[candidate_c[0]].append(j)  # add index s_j to cluster_i

    ##### UPDATE CENTROIDS WITH DBA #####
    centroids_update = np.zeros(centroids.shape)

    if verbose:
        print(f"{COL}[k-means]{CEND} Recalculating centroids...")

    # recalculate centroids
    for i in range(k):

        # collect all sequences of cluster_i in ndarray S_i
        L = len(clusters[i])
        assert L > 0  # FIX-ME: sometimes a cluster is empty--i.e. no sequences assigned to it.
        # if L == 0:
        #     print(clusters[i])

        S_i = np.zeros((L, S.shape[1]))

        for l, s_j in enumerate(clusters[i]):  # fetch s_j from list, add sequence to cluster set
            S_i[l] = S[s_j]

        # get updated sample mean of cluster_i
        dba = DBA(S_i, "VANILLA")
        centroids_update[i] = dba.mean()

    # return clusters, updated centroids
    return clusters, centroids_update


def main():
    # To run script and ensure import works:
    # python -m package.module
    # i.e. python -m src.kmeans
    from .dataset import Dataset

    ds = Dataset("Ham")
    S = ds.train_set
    k = ds.no_clusters

    # intialize k random centroids
    indices = np.random.randint(0, S.shape[0], k)
    C = np.zeros((k, S.shape[1]))
    for i in range(indices.size):
        C[i] = S[indices[i]]

    _, C_updated = k_means_iteration(C, S)

    j = 1
    while j < 3:
        C = np.copy(C_updated)
        _, C_updated = k_means_iteration(C, S)
        j += 1


if __name__ == "__main__":
    main()
