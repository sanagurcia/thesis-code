import os
from pathlib import Path
import numpy as np

# UCR Datasets: each line in a file contains a time series as list of CSVs
#  first entry of each sequences signifies target class (for clustering)

# GET N SEQUENCES FROM ALL CLASSES, AND CLASS INFORMATION
# @param n: number of wanted sequences in set
# @param dataset: UCR/<subpath>
# @param k: number of classes in set
def get_sequences(n: int, dataset: str, k: int) -> np.ndarray:
    # peek into file to figure out sequence length
    with open("UCR/" + dataset) as f:
        line = f.readline()
        seq = np.array(line.split(","), float)
        M = seq.size - 1  # first entry is the class information

    words_file = open("UCR/" + dataset)

    # initialize target array (rows=number of sequences, columns=points per sequence)
    Y = np.zeros((n, M))
    classes = [[] for i in range(k)]  # create list of k lists for each class

    for j in range(n):
        line = words_file.readline()  # read in one sequence
        seq = np.array(
            line.split(","), float
        )  # transform comma separated string into np array
        Y[j] = seq[1:]

        k = int(seq[0]) - 1
        classes[k].append(j)  # append S_j to corresponding class list

    return Y, classes


def get_class_sequences(n_seqs: int, dataset: str, k_classes: int) -> np.ndarray:
    """Return array of n same-class-sequences from dataset

    Args:
        n_seqs (int): no. of sequences
        dataset (str): absolute path to dataset
        k_classes (int): no. of classes in dataset

    Returns:
        np.ndarray: array of same-class-sequences arranged as rows
    """
    A = np.genfromtxt(dataset)  # read all data into ndarray
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


def get_dataset_path(name: str) -> str:
    """Get absolute path to dataset. Expects rootdir == thesis/code

    Args:
        name (str): directory name

    Returns:
        str: absolute path
    """
    data_relative_path = f"{name}/{name}_TEST.tsv"
    urc_archive_path = os.path.join(Path(os.getcwd()).parent, "data/UCRArchive_2018/")
    return os.path.join(urc_archive_path, data_relative_path)
