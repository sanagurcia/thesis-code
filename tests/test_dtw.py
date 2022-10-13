from src import DTW, ucrdata_util

DATASET = "Adiac"


def test_vanilla_dtw():
    S = ucrdata_util.get_class_sequences(2, DATASET)
    cost, path = DTW.dtw(S[0], S[1])
    # Cost is a positive number, warping path is at least as long as first sequence
    assert cost > 0 and path.shape[0] >= S[0].shape[0]
