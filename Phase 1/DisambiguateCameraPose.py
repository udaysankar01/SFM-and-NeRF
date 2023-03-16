import numpy as np

def DisambiguateCameraPose(Cset, Rset, Xset):
    """
    Find unique camera pose by checking the cheirality condition.

    Parameters
    ----------
    Cset : array-like
        the set of camera centers
    Rset : array-like
        the set of rotation matrices
    Xset : array-like
        set of vectors representing the 3D positions of points in space 
    
    Results
    -------
    best_C : array-like
        the corrected camera center
    best_R : array-like
        the corrected rotation matrix
    best_X : array-like
        the corrected set of vector representing the 3D positions of points in space 
    """
    best_C = None
    best_R = None
    best_X = None
    max_num_positive_depths = 0

    for C, R, X3d in zip(Cset, Rset, Xset):
        
        C = C.reshape(-1, 1)
        R3 = R[2].reshape(1, -1)
        x3d = X3d / X3d[:, 3].reshape(-1, 1)
        x3d = x3d[:,: 3]

        num_positive_depths = 0
        # R3.(X-C) and z values should be positive
        for x in x3d:
            x = x.reshape(-1, 1)
            z = x[2]
            if R3.dot(x - C) > 0 and z > 0:
                num_positive_depths += 1
        
        if num_positive_depths > max_num_positive_depths:
            best_C = C
            best_R = R
            best_X = X3d

    best_X = best_X / best_X[:, 3].reshape(-1, 1)
    
    return best_C, best_R, best_X