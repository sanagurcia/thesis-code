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
import time
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

    D = np.genfromtxt(get_dataset_path(dataset), dtype="float32")  # read raw data into A
    k = get_dataset_no_classes(dataset)

    seq_len = D[0].shape[0] - 1  # seq len == len of any seq in dataset minus class info
    n_seq = n if n > 0 else D.shape[0]  # no. of sequences as queried or default total

    S = np.zeros((n_seq, seq_len), dtype="float32")  # initialize target array
    classes = [[] for i in range(k)]  # create list of k lists for each class

    for i in range(n_seq):
        raw_seq = D[i]
        S[i] = raw_seq[1:]  # copy sequence to target array, excluding class info

        c_i = int(raw_seq[0]) - 1  # get class index, (transform to zero-indexed)
        classes[c_i].append(i)  # append s_i to corresponding class list

    return S, classes


def get_class_sequences(n_seqs: int, dataset: str, i_class=-1) -> np.ndarray:
    """Return array of n same-class-sequences from dataset

    Args:
        n_seqs (int): no. of sequences
        dataset (str): dataset dir name

    Returns:
        np.ndarray: array of same-class-sequences arranged as rows
    """

    dataset_path = get_dataset_path(dataset)
    k_classes = get_dataset_no_classes(dataset)

    A = np.genfromtxt(dataset_path, dtype="float32")  # read all data into ndarray
    seq_length = A[0].size - 1  # get sequence length
    S = np.zeros((n_seqs, seq_length), dtype="float32")  # init returned array S

    if i_class < 0:
        i_class = np.random.randint(1, k_classes)  # choose random class

    i = 0
    j = 0
    while i < n_seqs and j < A.shape[0]:  # either n seqs found or all seqs iterated
        seq = A[j]
        if int(seq[0]) == i_class:  # if sequences class member, add to S
            S[i] = seq[1:]
            i += 1
        j += 1

    return S[np.all(S, axis=1)]  # remove zero-valued rows


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


def plot_clusters(clusters: [np.ndarray], means: [np.ndarray]):
    """Plot list of clusters

    Args:
        clusters ([np.ndarray]): list of 2-D arrays
        means ([np.ndarray]): list of sequences
    """

    for c_i, cluster in enumerate(clusters):
        plot_cluster(cluster, means[c_i], c_i)


def plot_cluster(cluster: np.ndarray, mean: np.ndarray, c_n=0, dataset_name=""):
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
        title = f"Dataset '{dataset_name}', class with " + title
    else:
        title = f"Cluster no. {c_n} with " + title

    plt.title(title)

    # plot each sequence in cluster
    for s_i in range(cluster.shape[0]):
        plt.plot(range(seq_len), cluster[s_i], linewidth=0.5)

    # plot centroid black
    plt.plot(range(seq_len), mean, c="k", linewidth=1.5)


def extract_class_sequences(S: np.ndarray, classes: [[int]]) -> [np.ndarray]:
    """Return list of arrays, each containing sequences from one class

    Args:
        S (np.ndarray): all sequences
        classes ([[int]]): for each class, list of indices (corresponding to S)

    Returns:
        [np.ndarray]: list of arrays
    """
    c_sequences = []
    for c_indices in classes:
        # extract class sequences from entire set
        c_S = np.take(S, c_indices, axis=0)
        c_sequences.append(c_S)
    return c_sequences


def get_n_datasets(n=10, low_k=False) -> [str]:
    """Return name of n datasets

    Args:
        n (int, optional): Defaults to 10.

    Returns:
        [str]: list of n names
    """
    # fmt: off
    low_k_names = ["ArrowHead","BeetleFly","BirdChicken","CBF","ChlorineConcentration","Coffee","Computers","DistalPhalanxOutlineAgeGroup","DistalPhalanxOutlineCorrect","Earthquakes","ECG200","ECGFiveDays","FordA","FordB","GunPoint","Ham","HandOutlines","Herring","ItalyPowerDemand","LargeKitchenAppliances","Lightning2","Meat","MiddlePhalanxOutlineAgeGroup","MiddlePhalanxOutlineCorrect","MoteStrain","PhalangesOutlinesCorrect","ProximalPhalanxOutlineAgeGroup","ProximalPhalanxOutlineCorrect","RefrigerationDevices","ScreenType","ShapeletSim","SmallKitchenAppliances","SonyAIBORobotSurface1","SonyAIBORobotSurface2","StarLightCurves","Strawberry","ToeSegmentation1","ToeSegmentation2","TwoLeadECG","Wafer","Wine","WormsTwoClass","Yoga","BME","Chinatown","DodgerLoopGame","DodgerLoopWeekend","FreezerRegularTrain","FreezerSmallTrain","GunPointAgeSpan","GunPointMaleVersusFemale","GunPointOldVersusYoung","HouseTwenty","InsectEPGRegularTrain","InsectEPGSmallTrain","PowerCons","SemgHandGenderCh2","SmoothSubspace","UMD"]
    # fmt: off
    all_names = ["Adiac","ArrowHead","Beef","BeetleFly","BirdChicken","Car","CBF","ChlorineConcentration","CinCECGTorso","Coffee","Computers","CricketX","CricketY","CricketZ","DiatomSizeReduction","DistalPhalanxOutlineAgeGroup","DistalPhalanxOutlineCorrect","DistalPhalanxTW","Earthquakes","ECG200","ECG5000","ECGFiveDays","ElectricDevices","FaceAll","FaceFour","FacesUCR","FiftyWords","Fish","FordA","FordB","GunPoint","Ham","HandOutlines","Haptics","Herring","InlineSkate","InsectWingbeatSound","ItalyPowerDemand","LargeKitchenAppliances","Lightning2","Lightning7","Mallat","Meat","MedicalImages","MiddlePhalanxOutlineAgeGroup","MiddlePhalanxOutlineCorrect","MiddlePhalanxTW","MoteStrain","NonInvasiveFetalECGThorax1","NonInvasiveFetalECGThorax2","OliveOil","OSULeaf","PhalangesOutlinesCorrect","Phoneme","Plane","ProximalPhalanxOutlineAgeGroup","ProximalPhalanxOutlineCorrect","ProximalPhalanxTW","RefrigerationDevices","ScreenType","ShapeletSim","ShapesAll","SmallKitchenAppliances","SonyAIBORobotSurface1","SonyAIBORobotSurface2","StarLightCurves","Strawberry","SwedishLeaf","Symbols","SyntheticControl","ToeSegmentation1","ToeSegmentation2","Trace","TwoLeadECG","TwoPatterns","UWaveGestureLibraryAll","UWaveGestureLibraryX","UWaveGestureLibraryY","UWaveGestureLibraryZ","Wafer","Wine","WordSynonyms","Worms","WormsTwoClass","Yoga","ACSF1","AllGestureWiimoteX","AllGestureWiimoteY","AllGestureWiimoteZ","BME","Chinatown","Crop","DodgerLoopDay","DodgerLoopGame","DodgerLoopWeekend","EOGHorizontalSignal","EOGVerticalSignal","EthanolLevel","FreezerRegularTrain","FreezerSmallTrain","Fungi","GestureMidAirD1","GestureMidAirD2","GestureMidAirD3","GesturePebbleZ1","GesturePebbleZ2","GunPointAgeSpan","GunPointMaleVersusFemale","GunPointOldVersusYoung","HouseTwenty","InsectEPGRegularTrain","InsectEPGSmallTrain","MelbournePedestrian","MixedShapesRegularTrain","MixedShapesSmallTrain","PickupGestureWiimoteZ","PigAirwayPressure","PigArtPressure","PigCVP","PLAID","PowerCons","Rock","SemgHandGenderCh2","SemgHandMovementCh2","SemgHandSubjectCh2","ShakeGestureWiimoteZ","SmoothSubspace","UMD"]
    
    names = low_k_names if low_k else all_names
    names = np.asarray(names)   # allows indexing with array of random indices

    random_indices = np.random.randint(0, len(names), size=n)

    return names[random_indices]


def time_execution(foo, a, b):
    """Return execution time in milliseconds for function with two args.

    Args:
        foo (_type_): _description_
        a (_type_): _description_
        b (_type_): _description_

    Returns:
        float: execution duration
    """
    start = time.time()
    foo(a, b)
    end = time.time()
    interval = round((end - start) * 10**3, 2)
    return interval
