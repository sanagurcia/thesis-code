from src import ucrdata


def test_get_class_sequences():
    dataset_path = ucrdata.get_dataset_path("Adiac")
    S = ucrdata.get_class_sequences(10, dataset_path, 37)
    assert S.shape == (10, 176)
