import os
from pathlib import Path
import numpy as np

# type aliases
IndexList = [int]
ClusterIndexList = [IndexList]
ClusterSequenceList = [np.ndarray]

# global constants
DATASETS_SUMMARY_FILE = "clean_summary.tsv"

"""
This module assumes that the CWD has a sibling 'data' directory, containing the following:
- UCRArchive/
- DATASETS_SUMMARY_FILE

On UCR Dataset:
    Contains roughly 150 separate time-series datasets, each as individual directory.
    In each dataset directory 'UCRArchive_2018/<dataset>/':
        * <dataset>_TRAIN.tsv
        * <dataset>_TEST.tsv
            - .tsv's encode one sequence per line; first field (int) indicates class
        * README.md
            - info on no. of sequences, sequence length, and no. classes in dataset
"""


class Dataset:
    """A UCR dataset

    On init, reads in training and test sequences for given dataset name.
    Stores cluster labels and useful summary information.
    """

    name: str = ""
    train_set_size: int = 0
    test_set_size: int = 0
    no_clusters: int = 0
    sequence_length: int = 0

    train_set = np.ndarray
    test_set = np.ndarray
    train_labels: ClusterIndexList = []
    test_labels: ClusterIndexList = []

    train_clusters: ClusterSequenceList = []
    test_clusters: ClusterSequenceList = []

    # constructor
    def __init__(self, name: str = ""):
        self.name = name
        if name == "":
            self.name = Dataset.get_random_dataset_names(1)[0]

        self.__set_summary_data()  # set details like no. sequences in datasets, no. classes, sequence length
        self.__set_full_sequence_sets()  # set full train & test sequence datasets
        self.__set_cluster_sequences()  # extract & set sequences in cluster datasets
        self.print_summary()

    def __set_cluster_sequences(self):
        """Set lists of cluster sequences"""
        self.train_clusters = Dataset.get_cluster_sequence_list(self.train_set, self.train_labels)
        self.test_clusters = Dataset.get_cluster_sequence_list(self.test_set, self.test_labels)

    def __set_full_sequence_sets(self):
        """Get and set all sequences from dataset and cluster information"""

        S_train, train_labels = Dataset.get_all_sequences(self.name, self.no_clusters, train=True)
        S_test, test_labels = Dataset.get_all_sequences(self.name, self.no_clusters, train=False)

        # Sanity check: summary data == read-in data
        if self.sequence_length != -1:
            assert self.sequence_length == S_train[0].shape[0] == S_test[0].shape[0]
        assert self.train_set_size == S_train.shape[0]
        assert self.test_set_size == S_test.shape[0]

        self.train_set = S_train
        self.test_set = S_test
        self.train_labels = train_labels
        self.test_labels = test_labels

    def __set_summary_data(self):
        """Encapsulate statically getting summary data and setting to this object"""

        summary_dict = Dataset.get_dataset_summary(self.name)
        self.train_set_size = summary_dict["train_set_size"]
        self.test_set_size = summary_dict["test_set_size"]
        self.no_clusters = summary_dict["no_clusters"]
        self.sequence_length = summary_dict["sequence_length"]

    def print_summary(self):
        COL = "\033[92m"
        CEND = "\033[0m"
        print(
            f"\n{COL}{self.name} Dataset{CEND}\
            \nClusters: {self.no_clusters}\
            \nSequence length: {self.sequence_length}\
            \nTrain size: {self.train_set_size}\
            \nTest size: {self.test_set_size}\n"
        )

    @staticmethod
    def __get_data_dir_path() -> str:
        """Return path to parenty directory with UCR data."""

        return os.path.join(Path(os.getcwd()).parent, "data")

    @staticmethod
    def __get_dataset_path(name: str, train=True) -> str:
        """Return path to dataset."""

        subset = "TRAIN" if train else "TEST"
        return os.path.join(Dataset.__get_data_dir_path(), f"UCRArchive_2018/{name}/{name}_{subset}.tsv")

    @staticmethod
    def get_dataset_summary(name: str) -> dict:
        """Return summary information dictionary for given dataset name"""

        summary_path = os.path.join(Dataset.__get_data_dir_path(), DATASETS_SUMMARY_FILE)
        summary = np.genfromtxt(summary_path, delimiter=" ", dtype="str")

        for line in summary:
            # name, train size, test size, k-clusters, seq length
            if str(line[0]) == name:
                try:
                    seq_len = int(line[4])
                except ValueError:
                    seq_len = -1  # some datasets have "Vary"ing sequence length
                return {
                    "train_set_size": int(line[1]),
                    "test_set_size": int(line[2]),
                    "no_clusters": int(line[3]),
                    "sequence_length": seq_len,
                }

        raise RuntimeError(f"Dataset {name} not found.")

    @staticmethod
    def get_all_sequences(name: str, k_clusters: int, train=True) -> (np.ndarray, ClusterIndexList):
        """Get all sequences from dataset and sequence cluster information

        Args:
            name (str): dataset name
            train (bool): train or test set

        Returns:
            (np.ndarray, [[int]]): array with alls sequences as rows, cluster lists with sequence indices
        """

        D = np.genfromtxt(Dataset.__get_dataset_path(name, train), dtype="float32")  # read raw data into D

        seq_len = D[0].shape[0] - 1  # seq len == len of any seq in dataset minus cluster info
        n_seq = D.shape[0]  # no. of sequences

        S = np.zeros((n_seq, seq_len), dtype="float32")  # initialize target array
        clusters: ClusterIndexList = [[] for i in range(k_clusters)]  # create k lists for each cluster

        for i in range(n_seq):
            raw_seq = D[i]
            S[i] = raw_seq[1:]  # copy sequence to target array, excluding cluster info

            c_i = int(raw_seq[0]) - 1  # get cluster index, (transform to zero-indexed)
            clusters[c_i].append(i)  # append s_i to corresponding cluster list

        return S, clusters

    @staticmethod
    def get_cluster_sequence_list(S: np.ndarray, clusters: ClusterIndexList) -> ClusterSequenceList:
        """Create list of np.ndarrays, each containing all sequences in given cluster
        Args:
        S (np.ndarray): all sequences
        clusters (clusterList): for each cluster, list of indices (corresponding to S)

        Returns:
            [np.ndarray]: list of arrays
        """

        c_sequences = []
        for c_indices in clusters:
            # extract cluster sequences from entire set
            c_S = np.take(S, c_indices, axis=0)
            c_sequences.append(c_S)
        return c_sequences

    @staticmethod
    def get_random_dataset_names(n: int) -> [str]:
        # fmt: off
        names = ["Adiac","ArrowHead","Beef","BeetleFly","BirdChicken","Car","CBF","ChlorineConcentration","CinCECGTorso","Coffee","Computers","CricketX","CricketY","CricketZ","DiatomSizeReduction","DistalPhalanxOutlineAgeGroup","DistalPhalanxOutlineCorrect","DistalPhalanxTW","Earthquakes","ECG200","ECG5000","ECGFiveDays","ElectricDevices","FaceAll","FaceFour","FacesUCR","FiftyWords","Fish","FordA","FordB","GunPoint","Ham","HandOutlines","Haptics","Herring","InlineSkate","InsectWingbeatSound","ItalyPowerDemand","LargeKitchenAppliances","Lightning2","Lightning7","Mallat","Meat","MedicalImages","MiddlePhalanxOutlineAgeGroup","MiddlePhalanxOutlineCorrect","MiddlePhalanxTW","MoteStrain","NonInvasiveFetalECGThorax1","NonInvasiveFetalECGThorax2","OliveOil","OSULeaf","PhalangesOutlinesCorrect","Phoneme","Plane","ProximalPhalanxOutlineAgeGroup","ProximalPhalanxOutlineCorrect","ProximalPhalanxTW","RefrigerationDevices","ScreenType","ShapeletSim","ShapesAll","SmallKitchenAppliances","SonyAIBORobotSurface1","SonyAIBORobotSurface2","StarLightCurves","Strawberry","SwedishLeaf","Symbols","SyntheticControl","ToeSegmentation1","ToeSegmentation2","Trace","TwoLeadECG","TwoPatterns","UWaveGestureLibraryAll","UWaveGestureLibraryX","UWaveGestureLibraryY","UWaveGestureLibraryZ","Wafer","Wine","WordSynonyms","Worms","WormsTwoClass","Yoga","ACSF1","AllGestureWiimoteX","AllGestureWiimoteY","AllGestureWiimoteZ","BME","Chinatown","Crop","DodgerLoopDay","DodgerLoopGame","DodgerLoopWeekend","EOGHorizontalSignal","EOGVerticalSignal","EthanolLevel","FreezerRegularTrain","FreezerSmallTrain","Fungi","GestureMidAirD1","GestureMidAirD2","GestureMidAirD3","GesturePebbleZ1","GesturePebbleZ2","GunPointAgeSpan","GunPointMaleVersusFemale","GunPointOldVersusYoung","HouseTwenty","InsectEPGRegularTrain","InsectEPGSmallTrain","MelbournePedestrian","MixedShapesRegularTrain","MixedShapesSmallTrain","PickupGestureWiimoteZ","PigAirwayPressure","PigArtPressure","PigCVP","PLAID","PowerCons","Rock","SemgHandGenderCh2","SemgHandMovementCh2","SemgHandSubjectCh2","ShakeGestureWiimoteZ","SmoothSubspace","UMD"]
        names = np.asarray(names)   # allows indexing with array of random indices
        random_indices = np.random.randint(0, len(names), size=n)

        return names[random_indices]
