from src import utils, fast_dtw
from src.dataset import Dataset
from tslearn.metrics import dtw, dtw_path

# define globals: functions to be compared
# path fcns
F1 = fast_dtw.dtw_path
F2 = dtw_path

# cost fcns
G1 = fast_dtw.dtw_cost
G2 = dtw


def compare_speeds(f1, f2, f1_label, f2_label):

    ds = Dataset("Adiac")
    a = ds.train_set[0]
    b = ds.train_set[1]

    # calculate DTW cost for sanity check
    wrapped_f1 = utils.time_it(f1)
    wrapped_f2 = utils.time_it(f2)

    t1 = wrapped_f1(a, b)
    t2 = wrapped_f2(a, b)
    print(f"Time {f1_label}: {t1}ms, time {f2_label}: {t2}ms")


if __name__ == "__main__":
    compare_speeds(F1, F2, "sad dtw_path", "tslearn dtw_path")
    compare_speeds(G1, G2, "sad dtw_cost", "tslearn dtw_cost")
