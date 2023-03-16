import random
import numpy as np
from EstimateFundamentalMatrix import EstimateFundamentalMatrix

def GetInliersRANSAC(points1, points2, index):
    """
    Rejects the outliers from a set of feature matches and returns the inliner indices.

    Parameters
    ----------
    points1 : numpy.ndarray
        feature points for matching in first image
    points2 : numpy.ndarray
        feature points for matching in second image
    index : numpy.ndarray
        index of all feature matches

    Results
    -------
    inlier_index : numpy.ndarray
        index of all inlier feature matches
    outlier_index : numpy.ndarray
        index of all outlier feature matches for visulaization
    F_best : numpy.ndarray
        the best fundamental matrix
    """
    inlier_index = []
    M = 1000  # no of iterations
    eps = 0.2 # error threshold
    n = 0
    F_best = None

    for i in range(M):

        # select 8 correspondences randomly
        x1_hat = random.choices(points1, k=8)
        x2_hat = random.choices(points2, k=8)

        F = EstimateFundamentalMatrix(x1_hat, x2_hat)
        if F is None:
            continue

        S_index = []
        for j in range(len(points1)):
            
            x1j = np.array([points1[j, 0], points1[j, 1], 1])
            x2j = np.array([points2[j, 0], points2[j, 1], 1])

            error = np.abs(np.dot(np.transpose(x2j), np.dot(F, x1j)))
            # error = np.abs(np.transpose(x2j) @ F @ x1j)
            if error < eps:
                S_index.append(index[j])
        
        if n < len(S_index):
            n = len(S_index)
            inlier_index = S_index
            F_best = F
    
    outlier_index = np.setdiff1d(index, inlier_index)

    return inlier_index, outlier_index, F_best