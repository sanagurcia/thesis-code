"""Utils

This script reads in sequences from the UCR Dataset and transforms packages them into an ndarray.
Expects to be run from project rootdir, which is a sibling of data/UCRArchive_2018.

Useful functions:

    * get_class_sequences - returns ndarray of same-class-sequences from given dataset
    * plot_alignment - plots DTW alignment between 2 sequences
"""

import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# UCR Datasets: each line in a file contains a time series as list of CSVs
#  first entry of each sequences signifies target class (for clustering)

# GET N SEQUENCES FROM ALL CLASSES, AND CLASS INFORMATION
# NECESSARY FOR CLUSTERING
# @param n: number of wanted sequences in set
# @param dataset: UCR/<subpath>
# @param k: number of classes in set
# def get_sequences(n: int, dataset: str, k: int) -> np.ndarray:
#     # peek into file to figure out sequence length
#     with open("UCR/" + dataset) as f:
#         line = f.readline()
#         seq = np.array(line.split(","), float)
#         M = seq.size - 1  # first entry is the class information

#     words_file = open("UCR/" + dataset)

#     # initialize target array (rows=number of sequences, columns=points per sequence)
#     Y = np.zeros((n, M))
#     classes = [[] for i in range(k)]  # create list of k lists for each class

#     for j in range(n):
#         line = words_file.readline()  # read in one sequence
#         seq = np.array(
#             line.split(","), float
#         )  # transform comma separated string into np array
#         Y[j] = seq[1:]

#         k = int(seq[0]) - 1
#         classes[k].append(j)  # append S_j to corresponding class list

#     return Y, classes


def get_class_sequences(n_seqs: int, dataset: str) -> np.ndarray:
    """Return array of n same-class-sequences from dataset

    Args:
        n_seqs (int): no. of sequences
        dataset (str): dataset dir name

    Returns:
        np.ndarray: array of same-class-sequences arranged as rows
    """

    dataset_path = get_dataset_path(dataset)
    k_classes = get_dataset_no_classes(dataset)

    A = np.genfromtxt(dataset_path)  # read all data into ndarray
    seq_length = A[0].size - 1  # get sequence length
    S = np.zeros((n_seqs, seq_length))  # init returned array S

    n_class = np.random.randint(1, k_classes)  # choose random class

    i = 0
    j = 0
    while i < n_seqs and j < A.shape[0]:  # either n seqs found or all seqs iterated
        seq = A[j]
        if int(seq[0]) == n_class:  # if sequences class member, add to S
            S[i] = seq[1:]
            i += 1
        j += 1

    return S

def get_ucr_archive_path() -> str:
    """Return path to UCR Archive. Execution root must be parent of code/ & /data dirs.

    Returns:
        str: Absolute path to UCR Archive
    """
    return os.path.join(Path(os.getcwd()).parent, "data/UCRArchive_2018/")


def get_dataset_path(name: str) -> str:
    """Get absolute path to dataset.

    Args:
        name (str): directory name

    Returns:
        str: absolute path
    """
    data_relative_path = f"{name}/{name}_TEST.tsv"
    return os.path.join(get_ucr_archive_path(), data_relative_path)


def get_dataset_no_classes(name: str) -> int:
    """Get number of classes in dataset

    Args:
        name (str): dataset dir name

    Returns:
        int: no of classes
    """
    readme_path = f"{name}/README.md"
    absolute_path = os.path.join(get_ucr_archive_path(), readme_path)
    readme = open(absolute_path, encoding="utf-8")

    # Find line with number of classes
    while 1:
        line = readme.readline()
        if re.match("^Number of classses.*", line):
            res = re.findall("[0-9]+", line)
            return int(res[0])


def plot_alignment(path: np.ndarray, a: np.ndarray, b: np.ndarray, title="DTW Point-to-Point Alignment"):
    """Plot DTW alignemnt along warping path.
             
    Args:
        path (Nx2 Array): indices of warping path
        a (np.ndarray): first sequence
        b (np.ndarray): second sequence
        title (str, optional): title
    """

    plt.figure(figsize=(12, 5))  # set figure size very wide
    plt.title(title)

    for a_i, b_j in path:
        x_values = [a_i, b_j]
        y_values = [a[a_i], b[b_j] + 1]
        plt.plot(x_values, y_values, c="C7")

    # plot original curves (with displacement in second curve)
    plt.plot(range(a.shape[0]), a, "-o", c="g")  # '-o' means show pts
    plt.plot(range(b.shape[0]), b + 1, "-o", c="b")  # c is color, 'k' stands for black
