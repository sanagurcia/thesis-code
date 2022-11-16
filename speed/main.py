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


def main(f1, f2):

    ds = Dataset("Beef")
    S = ds.train_set

    a = S[0]
    b = S[1]

    path1 = f1(a, b)
    path2 = f2(a, b)

    print(path1)
    print(path2)

    # wrap functions with time_it


if __name__ == "__main__":
    main(F1, F2)
