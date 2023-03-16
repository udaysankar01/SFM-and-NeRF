import numpy as np
from scipy.optimize import least_squares


def NonLinearTriangulation(K, C1, R1, C2, R2, x1, x2, x0):
    """
    Computes the 3D position of a set of points given its projections in two images using
    non-linear triangulation.

    Parameters
    ----------
    K : array-like
        camera inrinsic matrix
    C1 : array-like
        center of first camera
    R1 : array-like
        rotation matrix of first camera
    C2 : array-like
        center of second camera
    R2 : array-like
        rotation matrix of second matrix
    x1 : array-like
        projections of a set of points in first image 
    x2 : array-like
        projections of a set of points in second image
    X : array-like
        a set of 3D points
    Results
    -------
    X : array-like
        linearly triangulated points
    """
    X = []
    I = np.identity(3)

    C1 = np.reshape(C1, (3, 1))        
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))

    C2 = np.reshape(C2, (3, 1))        
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    if len(x1) == len(x2) == len(x0):

        for i in range(len(x0)):
            optimal_values = least_squares(fun=Loss, x0=x0[i], method="trf", args=[x1[i], x2[i], P1, P2])
            X1 = optimal_values.x
            X.append(X1)
        
        X = np.array(X)
        X = X / X[:,3].reshape(-1,1)
    
    return X
    

def Loss(X, x1, x2, P1, P2):
    """
    Loss function for optimization in non-linear triangulation.

    Parameters
    ----------
    X : array-like
        linearly triangulated points
    x1 : array-like
        projecttion of a point in first image
    x2 : array-like
        projection of a point in the second image
    P1 : array-like
        first projection matrix
    P2 : array-like
        second projection matrix

    Results
    -------
    error : numpy.ndarray
        the error array calculated for optimization 
    """
    P1_1T, P1_2T, P1_3T = P1[0].reshape(1, -1), P1[1].reshape(1, -1), P1[2].reshape(1, -1)
    P2_1T, P2_2T, P2_3T = P2[0].reshape(1, -1), P2[1].reshape(1, -1), P2[2].reshape(1, -1)

    u1, v1 = x1
    u2, v2 = x2

    error1 = np.square(u1 - np.divide(np.dot(P1_1T, X), np.dot(P1_3T, X))) + np.square(v1 - np.divide(np.dot(P1_2T, X), np.dot(P1_3T, X)))
    error2 = np.square(u2 - np.divide(np.dot(P2_1T, X), np.dot(P2_3T, X))) + np.square(v2 - np.divide(np.dot(P2_2T, X), np.dot(P2_3T, X)))

    error = (error1 + error2).squeeze()

    return error