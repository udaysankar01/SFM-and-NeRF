import numpy as np

def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):
    """
    Computes the 3D position of a set of points given its projections in two images using
    linear triangulation.

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

    Results
    -------
    X : array-like
        set of vectors representing the 3D positions of points in space 
    """
    X = []

    C1 = np.reshape(C1, (3, 1))
    C2 = np.reshape(C2, (3, 1))

    P1 = np.dot(K, np.dot(R1, np.hstack((np.identity(3), -C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((np.identity(3), -C2))))

    p1T = P1[0].reshape(1, 4)
    p2T = P1[1].reshape(1, 4)
    p3T = P1[2].reshape(1, 4)

    p1T_dash = P2[0].reshape(1, 4)
    p2T_dash = P2[1].reshape(1, 4)
    p3T_dash = P2[2].reshape(1, 4)

    for i in range(x1.shape[0]):
        
        x, y = x1[i]
        x_dash, y_dash = x2[i]

        A1 = (y * p3T) - p2T
        A2 = p1T - (x * p3T)
        A3 = (y_dash * p3T_dash) - p2T_dash
        A4 = p1T_dash - (x_dash * p3T_dash)

        A = np.array([A1, A2, A3, A4]).reshape(4, 4)

        _, _, V = np.linalg.svd(A)
        V = np.transpose(V)
        x = V[:, -1]
        X.append(x)
    X = np.array(X)
    X = X / X[:, 3].reshape(-1, 1)

    return X


