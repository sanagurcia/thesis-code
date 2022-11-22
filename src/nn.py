from .fast_dtw import dtw_cost
import numpy as np

# type aliases
IndexList = [int]
ClusterIndexList = [IndexList]


def nearest_mean_classify(centroids: [np.ndarray], S: np.ndarray) -> ClusterIndexList:
    """Use nearest neighbor to associate each sequence in test set with closest mean.
    Complexity: O(n * k * m^2)

    Args:
        centroids ([np.ndarray]): list of class-mean sequences
        S (np.ndarray): sequencs to classify

    Returns:
        [[int]]: class lists with indices to sequences array
    """
    n = S.shape[0]
    k = len(centroids)

    clusters: ClusterIndexList = [[] for j in range(k)]

    # assign each sequence to cluster with closest centroid
    for j in range(n):  # for each sequence,
        candidate_c = (-1, np.Inf)  # candidate centroid: (index, cost)

        for i in range(k):  # for each centroid
            cost = dtw_cost(centroids[i], S[j])  # compute distance from seq to centroid

            if cost < candidate_c[1]:
                candidate_c = (i, cost)

        clusters[candidate_c[0]].append(j)  # add index s_j to cluster_i

    return clusters


def calculate_success_rate(given_labels: ClusterIndexList, calculated_labels: ClusterIndexList, n: int) -> float:

    # different approach: create sets of each cluster list and find difference
    total_diff = 0

    for i, _ in enumerate(given_labels):
        given_set = set(given_labels[i])
        calculated_set = set(calculated_labels[i])
        assert isinstance(given_set, set) and isinstance(calculated_set, set)
        total_diff += len(given_set.difference(calculated_set))

    wrong_ratio = total_diff / n
    return round(1 - wrong_ratio, 2)


if __name__ == "__main__":
    from src.dataset import Dataset
    from src import dba

    ds = Dataset()

    # compute cluster means with DBA
    cluster_means = []
    for cluster in ds.train_clusters:
        mean = dba.dba_mean(cluster)
        cluster_means.append(mean)

    # get TEST classes and classify with nearest neighbor method
    print(f"Performing mean-based NN on test set of length {ds.test_set_size}...")

    calculated_labels = nearest_mean_classify(cluster_means, ds.test_set)

    success = calculate_success_rate(ds.test_labels, calculated_labels, ds.test_set_size)
    print(f"Success rate: {success}")
