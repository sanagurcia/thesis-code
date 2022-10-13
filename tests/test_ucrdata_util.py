from src import ucrdata_util

DATASET = "Adiac"
DATASET_SEQ_LENGTH = 176


def test_get_class_sequences():
    S = ucrdata_util.get_class_sequences(10, DATASET)
    assert S.shape == (10, DATASET_SEQ_LENGTH)
