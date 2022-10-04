# Quick & dirty k-means implementation

import numpy as np
import mydtw
import mydba


# S: sequences data
# k: number of clusters
# return: k clusters with lists of indices, mean sequences for each cluster
def find_k_clusters(S: np.ndarray, k: int) -> (np.ndarray, np.ndarray):

    # intialize k random centroids
    indices = np.random.randint(0, S.shape[0], k)
    C = np.zeros((k, S.shape[1]))
    for i in range(indices.size):
        C[i] = S[indices[i]]

        
    clusters, C_updated = k_means_iteration(C, S)

    # while centroids don't converge, perform iteration
    while not np.allclose(C, C_updated):
                
        C = np.copy(C_updated)
        clusters, C_updated = k_means_iteration(C, S)
                
    return clusters, C_updated
    


# C: array of candidate centroids
# S: array of sequences (data points)
def k_means_iteration(C: np.ndarray, S: np.ndarray):

    N = C.shape[0]
    M = S.shape[0]

    # entry c_i in clusters is a list for indices s_j from S
    clusters = []
    for i in range(N):
        clusters.append([])
    
    # assign each sequence to cluster with closest centroid
    for j in range(M):				# for each sequence,
        candidate_c = (-1, np.inf)		# candidate centroid: (index, cost)
        
        for i in range(N):			# for each centroid
            cost, path = mydtw.ddtw(C[i], S[j])
	    
            if cost < candidate_c[1]:
                candidate_c = (i, cost)

        clusters[candidate_c[0]].append(j)	# add index s_j to cluster_i

        
    C_updated = np.zeros(C.shape)
    
    # recalculate centroids
    for i in range(N):

        # get all sequences in cluster_i
        L = len(clusters[i])
        S_i = np.zeros((L, S.shape[1]))	 
        
        for l, s_j in enumerate(clusters[i]):	# fetch s_j from list, add sequence to cluster set
            S_i[l] = S[s_j]

        # get updated sample mean of cluster_i, (3 iterations of alg.)
        C_updated[i] = mydba.dba_mean(S_i, 5)


    # return clusters, updated centroids
    return clusters, C_updated
            

'''
PSEUDO CODE:
Initialize k random centroids.
While centroids don't converge, perform iteration.

Iteration:
   for each datapoint:
       calculate distance to each centroid (DTW distance)
       assign sequence to nearest cluster

   recompute centroid, based on new datapoints (Time-Series averaging with DBA)

'''
