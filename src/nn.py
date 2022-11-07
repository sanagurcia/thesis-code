from .dba import dba_mean
from .fast_dtw import dtw_cost
import numpy as np


def test():
    from . import utils

    # dataset = utils.get_n_datasets(1)[0]
    dataset = "CBF"
    S_train, train_classes = utils.get_all_sequences(dataset)
    S_test, test_classes = utils.get_all_sequences(dataset, train=False)

    S_train_list = utils.extract_class_sequences(S_train, train_classes)

    class_means = []
    for S_c in S_train_list:
        mean = dba_mean(S_c)
        class_means.append(mean)

    print(
        f'Set "{dataset}"\nClasses: {len(train_classes)}\nSequences: {S_train.shape[0]}\nSequence length: {S_train.shape[1]}'
    )
    utils.plot_clusters(S_train_list, class_means)

    print(f"Performing NN on Test set of length {S_test.shape[0]}")

    calculated_classes = classify(class_means, S_test)  # do the meat

    S_test_list = utils.extract_class_sequences(S_test, calculated_classes)
    utils.plot_clusters(S_test_list, class_means)


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

    return correctly_classified / n


def get_class_for_sequence(class_list: [[int]], seq_index):
    for class_index, a_class in enumerate(class_list):
        if seq_index in a_class:
            return class_index
