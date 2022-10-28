from .dba import dba_mean
from .fast_dtw import dtw_cost


def test():
    from . import utils

    dataset = utils.get_n_datasets(1)[0]
