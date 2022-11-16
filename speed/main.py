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
    S = ds.train_set

    # warm up function
    print("Warming up functions")
    f1(S[0], S[1])
    f2(S[0], S[1])

    # wrap functions with time_it
    wf1 = utils.time_it(f1)
    wf2 = utils.time_it(f2)

    t1_total = 0
    t2_total = 0
    n = ds.train_set_size
    for i in range(n):
        t1_total += wf1(S[0], S[i])
        t2_total += wf2(S[0], S[i])

    t1_avg = round(t1_total / n, 2)
    t2_avg = round(t2_total / n, 2)

    print(f"{f1_label} average time: {t1_avg}")
    print(f"{f2_label} average time: {t2_avg}")
    print(f"speedup f2:f1 ~= {int((t1_avg/t2_avg)*100)}")


if __name__ == "__main__":
    compare_speeds(F1, F2, "sad dtw_path", "tslearn dtw_path")
    compare_speeds(G1, G2, "sad dtw_cost", "tslearn dtw_cost")
