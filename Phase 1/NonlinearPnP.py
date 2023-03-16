import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

def NonLinearPnP(X, x, K, C, R):
    """
    Non-linear Perspective-n-Point (PnP) algorithm for solving the pose estimation problem 
    using a set of 3D object points and their corresponding 2D image points.

    Parameters
    ----------
    X : numpy.ndarray
        a set of 3D points
    x : numpy.ndarray
        a set of projections of these 3D points
    K : numpy.ndarray
        camera intrinsic matrix
    C : numpy.ndarray
        the center of camera
    R : numpy.ndarray
        the rotation matrix
    
    Results
    -------
    Cnew : numpy.ndarray
        the estimated center of camera
    Rnew : numpy.ndarray
        the estimated rotation matrix
    """
    Q = Rotation.from_matrix(R).as_quat()

    X0 = [Q[0], Q[1], Q[2], Q[3], C[0], C[1], C[2]]

    optimized_values = least_squares(fun=NonLinearPnPLoss, x0=X0, method="trf", args=[X, x, K])
    X_opt = optimized_values.x

    Q = X_opt[: 4]
    Cnew = X_opt[4: ]
    Rnew = Rotation.from_quat(Q).as_matrix()

    return Cnew, Rnew


def NonLinearPnPLoss(X0, X, x, K):
    """
    The loss function for optimization in Non-Linear PnP. 

    Parameters
    ----------
    X0 : numpy.ndarray
        initial values of parameters for optimization
    X : numpy.ndarray
        a set of 3D points
    x : numpy.ndarray
        a set of projections of these 3D points
    K : numpy.ndarray
        camera intrinsic matrix

    Results
    -------
    error : numpy.ndarray
        the error for optimization in Non-Linear PnP
    """
    Q = X0[: 4]
    C = X0[4: ].reshape(-1, 1)
    R = Rotation.from_quat(Q).as_matrix()

    C_ = np.reshape(C, (3, 1))        
    P = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C_))))

    error_list = []

    for X_, x_ in zip(X, x):
        
        p_1T, p_2T, p_3T = P[0].reshape(1,-1), P[1].reshape(1,-1), P[2].reshape(1,-1)

        X_ = np.hstack((X_.reshape(1,-1), np.ones((X_.reshape(1,-1).shape[0], 1)))).reshape(-1, 1)

        u, v = x_[0], x_[1]
        e =  np.square(v - np.divide(np.dot(p_2T, X_) , np.dot(p_3T, X_))) + np.square(u - np.divide(np.dot(p_1T, X_), np.dot(p_3T, X_)))

        error_list.append(e)
    
    error_list = np.array(error_list)
    error = np.mean(error_list.squeeze())

    return error