import math
import numpy as np
from src.fast_dtw import multi_dtw_cost, multi_dtw_path
from src.fast_dtw import dtw_cost, dtw_path


DESCRIPTOR = "DERIVATIVE"
L = 5


def shapedtw_path(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    """Return shape dtw alignment between two sequences"""
    sd_a = to_shape_descriptors(seq_a)
    sd_b = to_shape_descriptors(seq_b)

    if len(sd_a.shape) == 1:  # if shape descriptors array 1D
        return dtw_path(sd_a, sd_b)

    return multi_dtw_path(sd_a, sd_b)


def shapedtw_cost(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Return shape dtw distance between two sequences"""
    sd_a = to_shape_descriptors(seq_a)
    sd_b = to_shape_descriptors(seq_b)

    if len(sd_a.shape) == 1:  # if shape descriptors array 1D
        return dtw_cost(sd_a, sd_b)

    return multi_dtw_cost(sd_a, sd_b)


def to_shape_descriptors(seq: np.ndarray) -> np.ndarray:
    """Return 2D shape descriptor mapping from given 1D sequence"""

    if DESCRIPTOR == "RAW_SUBSEQUENCE":
        return sample_subsequences(seq, L)

    if DESCRIPTOR == "DERIVATIVE":
        return derivative_descriptor(seq, L)


def derivative_descriptor(seq: np.ndarray, l: int) -> np.ndarray:
    """Return shape descriptors corresponding to DDTW"""

    subsequences = sample_subsequences(seq, l)
    descriptors = np.zeros(subsequences.shape[0], dtype="float32")  # 1D array
    for i in range(subsequences.shape[0]):
        # take difference between each two points in subsequence, average the differences
        descriptors[i] = np.mean(np.diff(subsequences[i]))

    return descriptors


def sample_subsequences(seq: np.ndarray, l: int) -> np.ndarray:
    """Return subsequence of length l centered on each point in seq."""
    m = seq.size
    assert l < m

    # pad sequence ends with repeated values
    p_seq = pad_ends(seq, l)

    # init subsequences array
    subsequences = np.zeros((m, l), dtype="float32")
    for i in range(m):
        subsequences[i] = p_seq[i : i + l]  # slice l-sized padded sequence

    return subsequences


def pad_ends(seq: np.ndarray, l: int) -> np.ndarray:
    """Repeat start and end values of given sequence"""
    pad_size = math.floor(l / 2)
    m = seq.size

    # pad right side
    right_values = [seq[m - 1]] * pad_size
    seq = np.append(seq, right_values)

    # pad left side
    left_values = [seq[0]] * pad_size
    return np.insert(seq, 0, left_values)


def main():
    a = np.arange(2, 12)
    sd = to_shape_descriptors(a)
    print(sd)


if __name__ == "__main__":
    main()
