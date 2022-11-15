"""Utils

This module reads in sequences from the UCR Dataset and packages them into an ndarray.
Expects to be run from project rootdir, which is a sibling of data/UCRArchive_2018.

On UCR Dataset:
    Contains roughly 150 separate time-series datasets, each as individual directory.
    In each dataset directory 'UCRARchive_2018/<dataset>/':
        * <dataset>_TRAIN.tsv
        * <dataset>_TEST.tsv
            - .tsv's encode one sequence per line; first field (int) indicates class
        * README.md
            - info on no. of sequences, sequence length, and no. classes in dataset

Useful functions:
    * get_n_sequences - returns ndarray of n sequences & class information
    * get_class_sequences - returns ndarray of same-class-sequences from given dataset
    * plot_alignment - plots DTW alignment between 2 sequences
"""

import time
import numpy as np
import matplotlib.pyplot as plt

# type aliases
IndexList = [int]
ClusterIndexList = [IndexList]
ClusterSequenceList = [np.ndarray]


def plot_alignment(path: np.ndarray, a: np.ndarray, b: np.ndarray, title="DTW Point-to-Point Alignment"):
    """Plot DTW alignemnt along warping path.

    Args:
        path (Nx2 Array): indices of warping path
        a (np.ndarray): first sequence
        b (np.ndarray): second sequence
        title (str, optional): title
    """

    plt.figure(figsize=(12, 5), dpi=400)  # set figure size very wide
    plt.title(title)

    for a_i, b_j in path:
        x_values = [a_i, b_j]
        y_values = [a[a_i], b[b_j] + 1]
        plt.plot(x_values, y_values, c="C7")

    # plot original curves (with displacement in second curve)
    plt.plot(range(a.shape[0]), a, "-o", c="g")  # '-o' means show pts
    plt.plot(range(b.shape[0]), b + 1, "-o", c="b")  # c is color, 'k' stands for black


def plot_clusters(clusters: ClusterSequenceList, means: [np.ndarray], dataset_name=""):
    """Plot list of clusters

    Args:
        clusters ([np.ndarray]): list of 2-D arrays
        means ([np.ndarray]): list of sequences
    """

    for c_i, cluster in enumerate(clusters):
        plot_cluster(cluster, means[c_i], c_i + 1, dataset_name)


def plot_cluster(cluster: np.ndarray, mean: np.ndarray, c_n=1, dataset_name=""):
    """Plot individual cluster

    Args:
        cluster (np.ndarray): 2-D array with sequences along rows
        mean (np.ndarray): mean sequence (i.e. centroid)
        c_n (int, optional): cluster number. Defaults to 0.
    """
    seq_len = cluster.shape[1]

    # Setup figure
    plt.figure(num=c_n, figsize=(12, 5), dpi=400)

    title = f"{len(cluster)} sequences"
    if dataset_name:
        title = f"Dataset '{dataset_name}', cluster no. {c_n} with " + title
    else:
        title = f"Cluster no. {c_n} with " + title

    plt.title(title)

    # plot each sequence in cluster
    for s_i in range(cluster.shape[0]):
        plt.plot(range(seq_len), cluster[s_i], linewidth=0.5)

    # plot centroid black
    plt.plot(range(seq_len), mean, c="k", linewidth=1.5)


def time_it(f):
    """Time execution duration of function; return milliseconds"""

    def wraped_f(*args, **kw):
        start = time.time()
        f(*args, **kw)
        end = time.time()
        return round((end - start) * 10**3, 3)  # in ms

    return wraped_f
