import numpy as np
from .fast_dtw import dtw_cost as vanilla_dtw_cost
from .shapedtw import shapedtw_cost


# type aliases
IndexList = [int]
ClusterIndexList = [IndexList]


class NN:
    cost_measure = vanilla_dtw_cost
    centroids = [np.ndarray]
    sequences = np.ndarray

    def __init__(self, sequences: np.ndarray, centroids: np.ndarray, measure: str):
        self.sequences = sequences
        self.centroids = centroids

        if measure == "VANILLA":
            self.cost_measure = vanilla_dtw_cost
        elif measure == "SHAPE":
            self.cost_measure = shapedtw_cost

    def classify_and_measure_success(self, given_labels: ClusterIndexList) -> float:
        """Predict labels via means-based NN & measure success rate compared to given labels."""

        predicted_labels = self.nearest_mean_classify()
        return calculate_success_rate(given_labels, predicted_labels, self.sequences.shape[0])

    def nearest_mean_classify(self) -> ClusterIndexList:
        """Use nearest neighbor to associate each sequence in test set with closest mean.
        Complexity: O(n * k * m^2)

        Args:
            centroids ([np.ndarray]): list of class-mean sequences
            S (np.ndarray): sequencs to classify

        Returns:
            [[int]]: class lists with indices to sequences array
        """
        n = self.sequences.shape[0]
        k = len(self.centroids)

        clusters: ClusterIndexList = [[] for j in range(k)]

        # assign each sequence to cluster with closest centroid
        for j in range(n):  # for each sequence,
            candidate_c = (-1, np.Inf)  # candidate centroid: (index, cost)

            for i in range(k):  # for each centroid
                cost = self.cost_measure(self.centroids[i], self.sequences[j])  # compute distance from seq to centroid

                if cost < candidate_c[1]:
                    candidate_c = (i, cost)

            clusters[candidate_c[0]].append(j)  # add index s_j to cluster_i

        return clusters


def calculate_success_rate(given_labels: ClusterIndexList, predicted_labels: ClusterIndexList, n: int) -> float:
    """Calculate difference between given labels & predicted labels. Divide difference--i.e.,
    wrongly classified instances--by size of set. Return success ratio.
    """

    # different approach: create sets of each cluster list and find difference
    total_diff = 0

    for i, _ in enumerate(given_labels):
        given_set = set(given_labels[i])
        predicted_set = set(predicted_labels[i])
        assert isinstance(given_set, set) and isinstance(predicted_set, set)
        total_diff += len(given_set.difference(predicted_set))

    wrong_ratio = total_diff / n
    return round(1 - wrong_ratio, 2)
