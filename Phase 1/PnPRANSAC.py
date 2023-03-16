import random
import numpy as np
from LinearPnP import LinearPnP


def linearPnPError(X, x, K, R, C):

    u, v = x
    X = np.hstack((x.reshape(-1, 1), np.ones((x.reshape(-1, 1).shape[0], 1)))).reshape(-1, 1)
    C = C.reshape(-1, 1)
    
    I = np.identity(3)
    C_ = np.reshape(C, (3, 1))        
    P = np.dot(K, np.dot(R, np.hstack((I, -C_))))

    P1, P2, P3 = P

    x_proj = np.hstack((np.divide(np.dot(P1, X), np.dot(P3, X)), np.divide(np.dot(P2, X), np.dot(P3, X))))
    x = np.hstack((u, v))
    e = np.linalg.norm(x - x_proj)
    
    return e  
    

def PnPRANSAC(X, x, K):
    """
    Estimates the 6-DoF camera pose with respect to a 3D object using the Perspective-n-Point (PnP) algorithm
    with Random Sample Consensus (RANSAC) to handle outliers.

    Parameters
    ----------
    X : numpy.ndarray
        a set of 3D points in the world
    x : numpy.ndarray
        the 2D projections of the 3D points in the image
    K : numpy.ndarray
        the camera intrinsic matrix

    Results
    -------
    Cnew : numpy.ndarray
        the estimated center of camera
    Rnew : numpy.ndarray
        the estimated rotaion matrix
    """
    M = 1000    # no of iterations
    eps = 5     # error threshold
    n = 0
    S_in = []
    Rnew = None
    Cnew = None

    for i in range(M):

        # select 6 random points
        Xset = np.array(random.choices(X, k=6))
        xset = np.array(random.choices(x, k=6))

        C, R = LinearPnP(Xset,xset, K)

        S_index = []

        if R is not None:

            for j in range(len(X)):
                x_j = x[j]
                X_j = X[j]
                e = linearPnPError(X_j, x_j, K, R, C)

                if e < eps:
                    S_index.append(j)
        
        if len(S_index) > n:
            n = len(S_index)
            Cnew = C
            Rnew = R


    return Cnew, Rnew