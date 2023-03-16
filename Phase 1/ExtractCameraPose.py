import numpy as np

def ExtractCameraPose(E):
    """
    Estimate the camera pose given the essentil matrix.

    Parameters
    ----------
    E : array-like
        The essential matrix
    
    Results
    -------
    Cset : array-like
        the set of camera centers
    Rset : array-like
        the set of rotation matrices
    """
    U, S, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R1 = np.dot(U, np.dot(W, V))
    R3 = np.dot(U, np.dot(np.transpose(W), V))
    C1 = U[:, 2]
    C2 = -U[:, 2]

    Rset = np.array([R1, R1, R3, R3])
    Cset = np.array([C1, C2, C1, C2])

    # correcting the camera pose if determinant of R is -1
    for i in range(4):
        if np.linalg.det(Rset[i]) < 0:
            Cset[i] = -Cset[i]
            Rset[i] = -Rset[i]

    return Cset, Rset