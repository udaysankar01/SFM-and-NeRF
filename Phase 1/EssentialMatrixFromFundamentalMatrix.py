import numpy as np

def EssentialMatrixFromFundamentalMatrix(F, K):
    """
    Computes the essential matrix from the given fundamental matrix and the camera internal matrix.

    Parameters
    ----------
    F : array-like
        The fundamental matrix
    K : array-like
        The camera internal matrix
    
    Results
    -------
    E : array-like
        The essential matrix
    """
    E = np.dot(np.transpose(K), np.dot(F, K))
    U, S, V = np.linalg.svd(E)
    # correcting 
    S = np.diag([1, 1, 0])
    E = np.dot(U, np.dot(S, V))

    return E