import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KDTree


def SIFTKDTreeMatcher(descriptor1, descriptor2, THRESH=0.49):
    '''
    SIFTSimpleMatcher 
    Match one set of SIFT descriptors (descriptor1) to another set of
    descriptors (decriptor2). Each descriptor from descriptor1 can at
    most be matched to one member of descriptor2, but descriptors from
    descriptor2 can be matched more than once.
    
    Matches are determined as follows:
    For each descriptor vector in descriptor1, find the Euclidean distance
    between it and each descriptor vector in descriptor2. If the smallest
    distance is less than thresh*(the next smallest distance), we say that
    the two vectors are a match, and we add the row [d1 index, d2 index] to
    the "match" array.
    
    INPUT:
    - descriptor1: N1 * 128 matrix, each row is a SIFT descriptor.
    - descriptor2: N2 * 128 matrix, each row is a SIFT descriptor.
    - thresh: a given threshold of ratio. Typically 0.7
    
    OUTPUT:
    - Match: N * 2 matrix, each row is a match. For example, Match[k, :] = [i, j] means i-th descriptor in
        descriptor1 is matched to j-th descriptor in descriptor2.
    '''
    match = []
    
    # Kd-tree initialization
    tree = KDTree(descriptor2)
    
    # Kd-tree query
    dist, ind = tree.query(descriptor1, k=2)
    for i in range(len(descriptor1)):
        if dist[i][0] < THRESH*dist[i][1]:
            match.append([i, ind[i][0]])
    
    return np.array(match)


# For testing
if __name__ == '__main__':
    data1 = np.array([[1, 1, 1],
                      [6, 6, 6]])
    data2 = np.array([[1, 1, 1],
                      [2, 2, 2],
                      [3, 3, 3],
                      [6, 6, 6]])
    match = SIFTKDTreeMatcher(data1, data2)
    print(match) # output must be [[0, 0], [1, 3]]