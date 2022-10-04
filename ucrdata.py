import numpy as np

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
    

# GET N SEQUENCES FROM SAME CLASS
# @param n: how many sequences
# @param dataset: path to set
# @param k: number of classes in dataset
def get_class_sequences(n: int, dataset: str, k: int) -> np.ndarray:

    # open file & setup Y with zeros
    with open('UCR/' + dataset) as f:
        line = f.readline()
        seq = np.array(line.split(','), float)
        M = seq.size -1 		

    words_file = open('UCR/' + dataset)
    Y = np.zeros((n, M))

    # choose random class
    n_class = np.random.randint(1, k)

    # add sequence to Y from class until all n sequences gathered
    i = 0
    while i < n:
        line = words_file.readline()
        seq = np.array(line.split(','), float)   
        if seq[0] == n_class:
            Y[i] = seq[1:]	   # if seq belongs to class, slice off the class and store
            i += 1
        
    return Y   
   
   
   
