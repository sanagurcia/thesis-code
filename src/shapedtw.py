import math
import numpy as np
from src.fast_dtw import multi_dtw_cost, multi_dtw_path


DESCRIPTOR = "IDENTITY"
L = 5


def path(seq_a: np.ndarray, seq_b: np.ndarray):
    """Return shape dtw alignment between two sequences"""
    sd_a = to_shape_descriptors(seq_a)
    sd_b = to_shape_descriptors(seq_b)

    return multi_dtw_path(sd_a, sd_b)


def cost(seq_a: np.ndarray, seq_b: np.ndarray):
    """Return shape dtw distance between two sequences"""
    sd_a = to_shape_descriptors(seq_a)
    sd_b = to_shape_descriptors(seq_b)

    return multi_dtw_cost(sd_a, sd_b)


def to_shape_descriptors(seq: np.ndarray) -> (np.ndarray):
    """Return 2D shape descriptor mapping from given 1D sequence"""

    subsequences = sample_subsequences(seq, L)

    return subsequences


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
    a = np.arange(10)
    s = sample_subsequences(a, 6)
    print(a)
    print(s)


if __name__ == "__main__":
    main()
