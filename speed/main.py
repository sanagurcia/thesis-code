from src import utils, fast_dtw
from src.Dataset import Dataset

# import dtw as g_dtw

# define globals: functions to be compared
F1 = fast_dtw.dtw_cost
# F2 = g_dtw.dtw


def compare_speeds():

    ds = Dataset("Adiac")
    a = ds.train_set[0]
    b = ds.train_set[1]

    # calculate DTW cost for sanity check
    c1 = F1(a, b)
    result = F2(a, b)
    c2 = result.distance
    print(f"Cost from F1: {round(c1, 2)}, cost from F2: {round(c2, 2)}")

    wrapped_F1 = utils.time_it(F1)
    wrapped_F2 = utils.time_it(F2)

    t1 = wrapped_F1(a, b)
    t2 = wrapped_F2(a, b)
    print(f"Time F1: {t1}ms, time F2: {t2}ms")


def test_fast_dtw():
    ds = Dataset("Adiac")
    a = ds.train_set[0]
    b = ds.train_set[1]

    cost = F1(a, b)
    print(f"cost: {cost}")
    wrapped_F1 = utils.time_it(F1)
    t1 = wrapped_F1(a, b)
    print(f"Time F1: {t1}ms")


if __name__ == "__main__":
    test_fast_dtw()
