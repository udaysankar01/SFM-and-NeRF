import numpy as np

def getIndexAndVisibilityMatrix(X_f, inlier_feature_flag, camIndex):
    """
    Get the indices of inliers and the visibility matrix.

    Parameters
    ----------
    X_f : numpy.ndarray
        the array of found 3D points
    inlier_feature_flag : numpy.ndarray
        the flag matrix representing the inliers
    camIndex : int
        the number of the image being registered
    
    Results
    -------
    X_index : numpy.ndarray
        indices of inliers
    visibility_matrix : numpy.ndarray

    """

    tempIndex = np.zeros((inlier_feature_flag.shape[0]), dtype = int)

    for i in range(camIndex + 1):
        tempIndex = tempIndex | inlier_feature_flag[:,i]

    X_index = np.where((X_f.reshape(-1)) & (tempIndex))
    
    visibility_matrix = X_f[X_index].reshape(-1,1)
    for n in range(camIndex + 1):
        visibility_matrix = np.hstack((visibility_matrix, inlier_feature_flag[X_index, n].reshape(-1,1)))

    return X_index, visibility_matrix[:, 1:]