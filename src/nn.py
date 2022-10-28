from .dba import dba_mean
from .fast_dtw import dtw_cost


def test():
    from . import utils

    dataset = utils.get_n_datasets(1)[0]
    S_all, classes = utils.get_all_sequences(dataset)

    [S_c] = utils.extract_class_sequences(S_all, classes)
