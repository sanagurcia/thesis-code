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

import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def get_n_sequences(dataset: str, n=-1) -> (np.ndarray, [[int]]):
    """Get n sequences from dataset and sequence class information

    Args:
        dataset (str): dataset name
        n (int): no. required sequences, defaults to total in dataset

    Returns:
        (sequences, [classes]): set of sequences, class lists with sequence indices
    """

    D = np.genfromtxt(get_dataset_path(dataset))  # read raw data into A
    k = get_dataset_no_classes(dataset)

    seq_len = D[0].shape[0] - 1  # seq len == len of any seq in dataset minus class info
    n_seq = n if n > 0 else D.shape[0]  # no. of sequences as queried or default total

    S = np.zeros((n_seq, seq_len))  # initialize target array
    classes = [[] for i in range(k)]  # create list of k lists for each class

    for i in range(n_seq):
        raw_seq = D[i]
        S[i] = raw_seq[1:]  # copy sequence to target array, excluding class info

        c_i = int(raw_seq[0]) - 1  # get class index, (transform to zero-indexed)
        classes[c_i].append(i)  # append s_i to corresponding class list

    return S, classes


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
    data_relative_path = f"{name}/{name}_TRAIN.tsv"
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


def plot_alignment(
    path: np.ndarray, a: np.ndarray, b: np.ndarray, title="DTW Point-to-Point Alignment"
):
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
