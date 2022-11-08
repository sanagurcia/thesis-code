from .fast_dtw import dtw_cost
import numpy as np


def classify(means: [np.ndarray], test_sequences: np.ndarray) -> [[int]]:
    """Use nearest neighbor to associate each sequence in test set with closest mean.
    Complexity: O(n * k * m^2)

    Args:
        means ([np.ndarray]): list of class-mean sequences
        test_sequences (np.ndarray): sequencs to classify

    Returns:
        [[int]]: class lists with indices to test_sequences array
    """
    n = test_sequences.shape[0]
    k = len(means)

    calculated_classes = [[] for j in range(k)]

    # for each sequence s_i in test set
    for i in range(n):
        s_i = test_sequences[i]
        closet_mean_index = -1
        lowest_cost = np.Inf

        # for each available class mean, calculate distance from s_i to class mean
        for j in range(k):
            cost = dtw_cost(s_i, means[j])
            if cost < lowest_cost:  # assign s_i to mean with lowest cost
                lowest_cost = cost
                closet_mean_index = j

        assigned_class = calculated_classes[closet_mean_index]
        assigned_class.append(i)  # assign i of s_i to closest mean list

    return calculated_classes


def calculate_success_rate(labeled_classes: [[int]], calculated_classes: [[int]], n: int) -> float:

    correctly_classified = 0

    for test_i in range(n):
        labeled_class = get_class_for_sequence(labeled_classes, test_i)
        calculated_class = get_class_for_sequence(calculated_classes, test_i)

        if labeled_class == calculated_class:
            correctly_classified += 1

    return round(correctly_classified / n, 2)


def get_class_for_sequence(class_list: [[int]], seq_index):
    for class_index, a_class in enumerate(class_list):
        if seq_index in a_class:
            return class_index
