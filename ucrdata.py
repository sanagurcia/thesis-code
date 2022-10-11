import numpy as np
import os
from pathlib import Path

# UCR Datasets: each line in a file contains a time series as list of CSVs
#  first entry of each sequences signifies target class (for clustering)

# GET N SEQUENCES FROM ALL CLASSES, AND CLASS INFORMATION
# @param n: number of wanted sequences in set
# @param dataset: UCR/<subpath>
# @param k: number of classes in set
def get_sequences(n: int, dataset: str, k: int) -> np.ndarray:
    # peek into file to figure out sequence length
    with open('UCR/' + dataset) as f:
        line = f.readline()
        seq = np.array(line.split(','), float)
        M = seq.size -1 		# first entry is the class information

        
    words_file = open('UCR/' + dataset)

    # initialize target array (rows=number of sequences, columns=points per sequence)
    Y = np.zeros((n, M))
    classes = [ [] for i in range(k) ]	# create list of k lists for each class
    
    for j in range(n):
        line = words_file.readline()              # read in one sequence
        seq = np.array(line.split(','), float)	# transform comma separated string into np array
        Y[j] = seq[1:]
    
        k = int(seq[0]) - 1
        classes[k].append(j)	# append S_j to corresponding class list

        
    return Y, classes
    

def get_class_sequences(n: int, dataset_path: str, k: int) -> np.ndarray:
    """Return array of n sequences of same class from dataset

    Args:
        n (int): no. of sequences
        dataset_path (str): <dir>/<file>
        k (int): no. of classes in dataset

    Returns:
        np.ndarray: array of sequences arranged as rows
    """
    A = np.genfromtxt(dataset_path)
    S = np.zeros((A.shape[0], A.shape[1] - 1))
    print(A.shape[1] - 1)
    # for i in range(A.shape[0]):
        

    return A[:,1:]


urc_archive_path = os.path.join(Path(os.getcwd()).parent, 'data/UCRArchive_2018/')
dataset_path = os.path.join(urc_archive_path, 'Adiac/Adiac_TEST.tsv')
data = get_class_sequences(10, dataset_path, 37)
print(data)