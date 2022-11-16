from src import utils, fast_dtw
from src.dataset import Dataset
from tslearn.metrics import dtw, dtw_path

# define globals: functions to be compared
F1 = fast_dtw.dtw_path
F2 = dtw_path


def compare_speeds():

    ds = Dataset("Adiac")
    a = ds.train_set[0]
    b = ds.train_set[1]

    p1 = F1(a, b)
    p2 = F2(a, b)
    print("path 1:")
    print(p1[:10])
    print("path 2:")
    print(p2[:10])

    # calculate DTW cost for sanity check
    wrapped_F1 = utils.time_it(F1)
    wrapped_F2 = utils.time_it(F2)

    t1 = wrapped_F1(a, b)
    t2 = wrapped_F2(a, b)
    print(f"Time F1: {t1}ms, time F2: {t2}ms")


def test_tslearn():
    ds = Dataset("Adiac")
    a = ds.train_set[0]
    b = ds.train_set[1]

    cost = F2(a, b)
    print(f"Tslearn dtw cost: {cost}")


if __name__ == "__main__":
    compare_speeds()
