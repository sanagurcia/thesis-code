import os
from pathlib import Path
import numpy as np

# type aliases
IndexList = [int]
ClassList = [IndexList]

# global constants
DATASETS_SUMMARY_FILE = "clean_summary.tsv"


class Dataset:
    """A UCR dataset

    On init, reads in training and test sequences for given dataset name.
    Stores class labels and useful summary information.
    """

    name: str = ""
    train_set_size: int = 0
    test_set_size: int = 0
    no_classes: int = 0
    sequence_length: int = 0

    train_set = np.ndarray
    test_set = np.ndarray
    train_labels: ClassList = []
    test_labels: ClassList = []

    # constructor
    def __init__(self, name: str):
        # set summary information
        summary_dict = Dataset.get_dataset_summary(name)
        self.name = name
        self.train_set_size = summary_dict["train_set_size"]
        self.test_set_size = summary_dict["test_set_size"]
        self.no_classes = summary_dict["no_classes"]
        self.sequence_length = summary_dict["sequence_length"]

        # set train & test sequence sets
        self.get_sequences(train=True)
        self.get_sequences(train=False)

    def get_sequences(self, train=True) -> (np.ndarray, ClassList):
        """Get all sequences from dataset and sequence class information

        Args:
            train (bool): train set or test set

        Returns:
            (sequences, ClassList): set of sequences, class lists with sequence indices
        """

        D = np.genfromtxt(Dataset.get_dataset_path(self.name, train), dtype="float32")  # read raw data into array

        seq_len = D[0].shape[0] - 1  # seq len == len of any seq in dataset minus class info
        n_seq = D.shape[0]

        # Sanity check: summary data == read-in data
        assert self.sequence_length == seq_len
        if train:
            assert self.train_set_size == n_seq
        else:
            assert self.test_set_size == n_seq

        S = np.zeros((n_seq, seq_len), dtype="float32")  # initialize target array
        classes: ClassList = [[] for i in range(self.no_classes)]  # create index list for each class

        for i in range(n_seq):
            raw_seq = D[i]
            S[i] = raw_seq[1:]  # copy sequence to target array, excluding class info

            c_i = int(raw_seq[0]) - 1  # get class index, (transform to zero-indexed)
            classes[c_i].append(i)  # append s_i to corresponding class list

        # init sequence-set & class-labels
        if train:
            self.train_set = S
            self.train_labels = classes
        else:
            self.test_set = S
            self.test_labels = classes

    @staticmethod
    def get_data_dir_path() -> str:
        """Return path to parenty directory with UCR data."""

        return os.path.join(Path(os.getcwd()).parent, "data")

    @staticmethod
    def get_dataset_path(name: str, train=True) -> str:
        """Return path to dataset."""

        subset = "TRAIN" if train else "TEST"
        return os.path.join(Dataset.get_data_dir_path(), f"UCRArchive_2018/{name}/{name}_{subset}.tsv")

    @staticmethod
    def get_dataset_summary(name: str) -> dict:
        """Return summary information dictionary for given dataset name"""

        summary_path = os.path.join(Dataset.get_data_dir_path(), DATASETS_SUMMARY_FILE)
        summary = np.genfromtxt(summary_path, delimiter=" ", dtype="str")

        for line in summary:
            # name, train size, test size, k-classes, seq length
            if str(line[0]) == name:
                return {
                    "train_set_size": int(line[1]),
                    "test_set_size": int(line[2]),
                    "no_classes": int(line[3]),
                    "sequence_length": int(line[4]),
                }

        raise RuntimeError(f"Dataset {name} not found.")
