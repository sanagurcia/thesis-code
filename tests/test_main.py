from src import fast_dtw, dba, kmeans, utils


DATASET = "Adiac"
DATASET_SEQ_LENGTH = 176


def test_get_class_sequences():
    S = utils.get_class_sequences(10, DATASET)
    assert S.shape == (10, DATASET_SEQ_LENGTH)


def test_get_n_sequences():
    S, classes = utils.get_n_sequences(DATASET, 100)
    assert S.shape == (100, DATASET_SEQ_LENGTH)
    assert len(classes) == utils.get_dataset_no_classes(DATASET)


def test_dtw():
    S = utils.get_class_sequences(2, DATASET)
    cost = fast_dtw.dtw_cost(S[0], S[1])
    path = fast_dtw.dtw_path(S[0], S[1])

    # Cost is a positive number, warping path is at least as long as first sequence
    assert cost > 0 and path.shape[0] >= S[0].shape[0]


def test_dba():
    S = utils.get_class_sequences(10, DATASET)  # Copute mean for 10 sequences
    mean = dba.dba_mean(S, 3, verbose=True)  # Perform 3 iterations

    # Compare cost from old mean to new mean
    old_cost = dba.calculate_average_cost_to_mean(S[0], S)
    new_cost = dba.calculate_average_cost_to_mean(mean, S)

    assert new_cost < old_cost
    print(f"Starting average cost: {round(old_cost, 2)}\nEnding average cost: {round(new_cost, 2)}")


def test_kmeans():
    S, classes = utils.get_n_sequences("CBF")
    no_classes = utils.get_dataset_no_classes("CBF")
    clusters, centroids = kmeans.find_k_clusters(S, no_classes, True)
    assert centroids.shape[0] == no_classes
