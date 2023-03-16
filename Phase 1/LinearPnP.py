import numpy as np

def LinearPnP(Xset, xset, K):
    """
    Estimates camera pose using linear least squares method on the 3D points and corresponding 2D projection on
    the image.

    Parameters
    ----------
    Xset : numpy.ndarray
        set of 3D points
    xset : numpy.ndarray
        set of 2D projections of 3D points on the image
    K : numpy.ndarray
        camera intrinsic matrix
    
    Results
    -------
    C : numpy.ndarray
        the estimated center of camera
    R : numpy.ndarray
        the estimated rotation matrix
    """

    X4 = np.hstack((Xset, np.ones((Xset.shape[0], 1))))
    x3 = np.hstack((xset, np.ones((xset.shape[0], 1))))

    inv_K = np.linalg.inv(K)
    xn = np.transpose(np.dot(inv_K, x3.T))

    for i in range(Xset.shape[0]):

        X = X4[i].reshape((1, 4))
        
        u, v = xn[i, :2]

        u_cross = np.array([[0, -1, v],
                            [1, 0, -u],
                            [-v, u, 0]])
        Z = np.zeros((1, 4))
        X_hat = np.vstack((np.hstack((X, Z, Z)),
                           np.hstack((Z, X, Z)),
                           np.hstack((Z, Z, X))))
        a = np.dot(u_cross, X_hat)

        if i == 0:
            A = a
        else: 
            A = np.vstack((A, a))

    U, S, V = np.linalg.svd(A)
    P = V[-1].reshape((3, 4))
    R = P[:, : 3]
    
    # enforcing orthonormality
    Ur, S, Vr = np.linalg.svd(R)
    R = np.dot(Ur, Vr)

    C = P[:, 3]
    C = -np.dot(np.linalg.inv(R), C) 

    # correct the determinant
    if np.linalg.det(R) < 0:
        C = -C
        R = -R
    
    return C, R