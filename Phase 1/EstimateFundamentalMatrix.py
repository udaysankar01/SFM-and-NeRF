import numpy as np

def EstimateFundamentalMatrix(points1, points2):
    """
    Estimates the fundamental matrix from the eight randomly selected feature matches.

    Parameters
    ----------
    points1 : array-like
        points for matching from image 1 
    points2 : array-like
        points for matching from image 2

    Results
    -------
    F : array-like
         the resulting fundamental matrix
    """
    # normalize the points to improve numerical stability
    pts1_norm = (points1 - np.mean(points1, axis=0)) / np.std(points1, axis=0)
    pts2_norm = (points1 - np.mean(points2, axis=0)) / np.std(points2, axis=0)

    # construct the A matrix
    A = np.zeros((len(points1), 9))

    for i in range(len(points1)):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]
            
    # using Singular Value Decomposition for finding the Fundamental matrix
    U, S, V = np.linalg.svd(A)
    F_rank3 = V[-1].reshape(3, 3)
    
    # enforce rank-2 constraint on F
    U, S, V = np.linalg.svd(F_rank3)
    S[-1] = 0
    F_rank2 = np.dot(U, np.dot(np.diag(S), V))

    # denormalize the fundamental matrix 
    s1 = np.std(points1, axis=0)
    s2 = np.std(points2, axis=0)

    T1 = np.array([[1/s1[0], 0, ]])
    T1 = np.array([[1/s1[0], 0, -np.mean(points1, axis=0)[0]/s1[0]],
                   [0, 1/s1[1], -np.mean(points1, axis=0)[1]/s1[1]],
                   [0, 0, 1]])
    T2 = np.array([[1/s2[0], 0, -np.mean(points2, axis=0)[0]/s2[0]],
                   [0, 1/s2[1], -np.mean(points2, axis=0)[1]/s2[1]],
                   [0, 0, 1]])
    
    F = np.dot(np.transpose(T2), np.dot(F_rank2, T1))

    return F
