from src import ucrdata

DATASET = "Adiac"
DATASET_SEQ_LENGTH = 176


def test_get_class_sequences():
    dataset_path = ucrdata.get_dataset_path(DATASET)
    n_classes = ucrdata.get_dataset_no_classes(DATASET)
    S = ucrdata.get_class_sequences(10, dataset_path, n_classes)
    assert S.shape == (10, DATASET_SEQ_LENGTH)
