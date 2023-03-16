import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from PnPRANSAC import PnPRANSAC
from BundleAdjustment import BundleAdjustment
from NonlinearPnP import NonLinearPnP
from GetInliersRANSAC import GetInliersRANSAC
from ExtractCameraPose import ExtractCameraPose
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from LinearTriangulation import LinearTriangulation
from NonLinearTriangulation import NonLinearTriangulation
from DisambiguateCameraPose import DisambiguateCameraPose
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from helperFunctions import readImages, getMatchingFeatures, displayMatches, displayMatchesAfterRANSAC, visualizeEpipoles
from helperFunctions import meanReprojError, visualize2DPlot, ReprojErrorPnP

debug = True

def main():
    
    print('\n')
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default="./Data", help='the path of folder containing the image files')
    Parser.add_argument('--ResultsPath', default='./Results', help='the path of folder where results are to be stored')

    Args = Parser.parse_args()
    data_path = Args.DataPath
    results_path = Args.ResultsPath
    debug_path = './Data/IntermediateOutputImages'
    use_saved = True

    # check if results folder already exists
    if not os.path.exists(results_path):
        print('Results path not found! Creating folder...')
        os.makedirs(results_path)
    
    images = readImages(data_path)
    
    if debug:
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)
        for i, image in enumerate(images):
            cv2.imwrite(f'{debug_path}/{i+1}.png', image)


    """
    Getting the feature matchings from teh given matching text files.
    """

    features_u, features_v, flag_feature = getMatchingFeatures(data_path, images)
    

    """
    Reject the outliers using RANSAC.
    """
    if use_saved:
        print("\nSkipping RANSAC and using saved feature data.\n")
        inlier_feature_flag = np.load('./SavedData/inlier_feature_flag.npy', allow_pickle=True)
        F_array = np.load('./SavedData/Farray.npy', allow_pickle=True)
        # inlier_points1_array = np.load('./SavedData/inlier_points1_array.npy', allow_pickle=True)
        # inlier_points2_array = np.load('./SavedData/inlier_points2_array.npy', allow_pickle=True)

    else:
        print("\nInitiating RANSAC outlier rejection and computing Fundamental matrix...\n")
        F_array = np.empty(shape=(images.shape[0], images.shape[0]), dtype=object)
        inlier_feature_flag = np.zeros_like(flag_feature)
        
        for i in range(images.shape[0]- 1):
            
            for j in range(i + 1, images.shape[0]):

                index = np.where(flag_feature[:, i] & flag_feature[:, j])
                pts1 = np.hstack((features_u[index, i].reshape((-1, 1)), features_v[index, i].reshape((-1, 1))))
                pts2 = np.hstack((features_u[index, j].reshape((-1, 1)), features_v[index, j].reshape((-1, 1))))

                if debug: 
                    displayMatches(images[i], images[j], pts1, pts2, file_path=f'{debug_path}/matching{i+1}{j+1}.png')

                index = np.reshape(np.array(index), -1)
                if len(index) > 8:
                    inlier_index, outlier_index, F = GetInliersRANSAC(pts1, pts2, index)
                    print(f'Images {i+1} and {j+1}: {len(inlier_index)} inliers out of {len(index)} features.\n')
                    # F_array[i, j] = F
                    inlier_feature_flag[inlier_index, i] = 1
                    inlier_feature_flag[inlier_index, j] = 1
                
                # if debug:
                #     inlier_pts1 = np.hstack((features_u[inlier_index, i].reshape((-1, 1)), features_v[inlier_index, i].reshape((-1, 1))))
                #     inlier_pts2 = np.hstack((features_u[inlier_index, j].reshape((-1, 1)), features_v[inlier_index, j].reshape((-1, 1))))
                #     inlier_points1_array[i][j] = inlier_pts1
                #     inlier_points2_array[i][j] = inlier_pts2
                #     outlier_pts1 = np.hstack((features_u[outlier_index, i].reshape((-1, 1)), features_v[outlier_index, i].reshape((-1, 1))))
                #     outlier_pts2 = np.hstack((features_u[outlier_index, j].reshape((-1, 1)), features_v[outlier_index, j].reshape((-1, 1))))
                #     displayMatchesAfterRANSAC(images[i], images[j], inlier_pts1, inlier_pts2, outlier_pts1, outlier_pts2, file_path=f'{debug_path}/after_RANSAC{i+1}{j+1}.png')

        
    """
    Compute the fundamental matrix.
    """
    for i in range(images.shape[0]- 1):
            
        for j in range(i + 1, images.shape[0]):

            inlier_index = np.where(inlier_feature_flag[:,i] & inlier_feature_flag[:,j])
            inlier_pts1 = np.hstack((features_u[inlier_index, i].reshape((-1, 1)), features_v[inlier_index, i].reshape((-1, 1))))
            inlier_pts2 = np.hstack((features_u[inlier_index, j].reshape((-1, 1)), features_v[inlier_index, j].reshape((-1, 1))))

            F = EstimateFundamentalMatrix(inlier_pts1, inlier_pts2)
            F_array[i][j] = F
            # F = F_array[i][j]

            if debug:
                outlier_index = np.where(~(inlier_feature_flag[:,i] & inlier_feature_flag[:,j]))
                outlier_pts1 = np.hstack((features_u[outlier_index, i].reshape((-1, 1)), features_v[outlier_index, i].reshape((-1, 1))))
                outlier_pts2 = np.hstack((features_u[outlier_index, j].reshape((-1, 1)), features_v[outlier_index, j].reshape((-1, 1))))
                displayMatchesAfterRANSAC(images[i], images[j], inlier_pts1, inlier_pts2, outlier_pts1, outlier_pts2, file_path=f'{debug_path}/after_RANSAC{i+1}{j+1}.png')
                visualizeEpipoles(images[i], images[j], inlier_pts1, inlier_pts2, F, filename1=f'{debug_path}/epipoles_F{i+1}{j+1}-{i+1}.png', filename2=f'{debug_path}/epipoles_F{i+1}{j+1}-{j+1}.png')



    # to save the inlier features and the fundamental matrices
    if not use_saved:
        if not os.path.exists('./SavedData'):
            os.makedirs('./SavedData')
        with open('./SavedData/inlier_feature_flag.npy', 'wb') as f:
            np.save(f, inlier_feature_flag)
        with open('./SavedData/Farray.npy', 'wb') as f:
            np.save(f, F_array)



    """
    For first and second image with inliers.    
    """
    print("######################################################################\n")
    print("Registering first and second cameras...\n")
    i, j = 0, 1
    F = F_array[i, j]
    print(f"The fundamental matrix for images 1 and 2:\n{F}\n")
    
    # read the camera internal matrix K from 'calibration.txt' file
    K = np.zeros((3, 3))
    f = open(f'{data_path}/calibration.txt', 'r')
    for i, row in enumerate(f):
        K[i] = np.float32(np.array(row.split()))

    """
    Estimating the Essential Matrix from the Fundamental Matrix.
    """
    print("Estimating the Essential Matrix from the Fundamental Matrix...\n")
    E = EssentialMatrixFromFundamentalMatrix(F, K)
    print(f"The essential matrix for images 1 and 2: \n{E}\n")

    print('Estimating Camera Pose...\n')
    Cset, Rset = ExtractCameraPose(E)
    
    index = np.where(inlier_feature_flag[:,i] & inlier_feature_flag[:,j])
    x1 = np.hstack((features_u[index, i].reshape((-1, 1)), features_v[index, i].reshape((-1, 1))))
    x2 = np.hstack((features_u[index, j].reshape((-1, 1)), features_v[index, j].reshape((-1, 1))))

    """
    Perform linear triangulation.
    """
    print("Performing Linear Triangulation...\n")
    pts3D_h = []
    C1 = np.zeros((3, 1))
    R1 = np.identity(3)
    for i in range(len(Cset)):
        X = LinearTriangulation(K, C1, R1, Cset[i], Rset[i], x1, x2)
        pts3D_h.append(X)
    print("Linear Triangulation Complete.\n")

    """
    Check the cheirality condition.
    """
    print("Disambiguating the camera poses...\n")
    C, R, X0 = DisambiguateCameraPose(Cset, Rset, pts3D_h)

    """
    # Perform non-linear triangulation.
    # """
    print("Performing Non-Linear Triangulation...\n")
    C1 = np.zeros((3, 1))
    R1 = np.identity(3)
    X_optimized = NonLinearTriangulation(K, C1, R1, C, R, x1, x2, X0)
    print("Non-Linear Triangulation complete.\n")

    meanReprojError1 = meanReprojError(X, x1, x2, R1, C1, R, C, K)
    meanReprojError2 = meanReprojError(X_optimized, x1, x2, R1, C1, R, C, K)

    print(f"Error before Non-Linear Triangulation : {meanReprojError1}")
    print(f"Error after Non-Linear Triangulation : {meanReprojError2}\n")

    X_complete = np.zeros((len(features_u), 3))
    camera_index = np.zeros((len(features_u), 1), dtype=int)
    X_f = np.zeros((len(features_u), 1), dtype=int)

    X_complete[index] = X[:, :3]
    camera_index[index] = 1
    X_f[index] = 1

    # setting points below origin along z-axis as zero
    X_f[np.where(X_complete[:, 2] < 0)] = 0

    Cset_ = [np.zeros(3), C]
    Rset_ = [np.identity(3), R]


    print("Cameras 1 and 2 registered!")
    print("######################################################################\n")

    # visualize2DPlot(X, Cset_, Rset_, filename=f'{debug_path}/topView12.png')    
    """
    Register camera and add 3D points for rest of the images.
    """

    print("Registering remaining cameras...\n")
    for i in range(2, images.shape[0]):

        # register ith image using PnP
        print("\n######################################################################\n")
        print(f"Registering image {i+1}...\n")

        feature_index_i = np.where(X_f[:, 0] & inlier_feature_flag[:, i])
        # print(f"\n{feature_index_i}\n")
        if len(feature_index_i[0]) < 8:
            print(f"Number of common points between X and image {i+1} : {len(feature_index_i)}\n")
            continue
        
        x = np.hstack((features_u[feature_index_i, i].reshape(-1, 1), features_v[feature_index_i, i].reshape(-1, 1)))
        X = X_complete[feature_index_i, :].reshape(-1, 3)

        """
        Perform PnP RANSAC.
        """
        print("Performing PnP-RANSAC...\n")
        Cnew, Rnew = PnPRANSAC(X, x, K)
        print("PnP-RANSAC complete!\n")
        # print(f"Cnew :\n{Cnew}")
        # print(f"Rnew :\n{Rnew}")
        linearPnPError = ReprojErrorPnP(X, x, K, Rnew, Cnew)
        

        """
        Perform Non-Linear PnP RANSAC.
        """
        print("Performing Non-Linear PnP...\n")
        Cnew, Rnew = NonLinearPnP(X, x, K, Cnew, Rnew)
        print("Non-Linear PnP complete!\n")
        # print(f"Cnew :\n{Cnew}")
        # print(f"Rnew :\n{Rnew}")
        nonLinearPnPError = ReprojErrorPnP(X, x, K, Rnew, Cnew)
        print(f"Error after linear PnP : {linearPnPError}")
        print(f"Error after non-linear PnP : {nonLinearPnPError}\n")

        Cset_.append(Cnew)
        Rset_.append(Rnew)

        """
        Perform Linear and Non-Linear Triangulation.
        """
        for j in range(i):
            
            X_index = np.where(inlier_feature_flag[:, i] & inlier_feature_flag[:, j])
            if (len(X_index[0])) < 8:
                continue
            
            x1 = np.hstack((features_u[X_index, j].reshape((-1, 1)), features_v[X_index, j].reshape((-1, 1))))
            x2 = np.hstack((features_u[X_index, i].reshape((-1, 1)), features_v[X_index, i].reshape((-1, 1))))


            """
            Perform Linear Triangulation
            """
            print("Performing Linear Triangulation...\n")
            Xnew = LinearTriangulation(K, Cset_[j], Rset_[j], Cnew, Rnew, x1, x2)
            print("Linear Triangulation complete!\n")
            LinTriError = meanReprojError(Xnew, x1, x2, Rset_[j], Cset_[j], Rnew, Cnew, K)


            """
            Perform Non-Linear Triangulation
            """
            print("Performing Non-Linear Triangulation...\n")
            Xnew = NonLinearTriangulation(K, Cset_[j], Rset_[j], Cnew, Rnew, x1, x2, Xnew)
            print("Non-Linear Triangulation complete!\n")
            NonLinTriError = meanReprojError(Xnew, x1, x2, Rset_[j], Cset_[j], Rnew, Cnew, K)
            
            print(f"Error after Linear Triangulation: {LinTriError}")
            print(f"Error after Non-Linear Triangulation: {NonLinTriError}")

            X_complete[X_index] = Xnew[:, : 3]
            X_f[X_index] = 1
            print(f"No of points between {j+1} and {i+1}: {len(X_index[0])}")

        # plotting the top view
        plot_index = np.where(X_f[:, 0])
        X = X_complete[plot_index]
        visualize2DPlot(X, Cset_, Rset_, filename=f'{debug_path}/topView{i}{j}.png')


    
        """
        Build Visibility Matrix and Perform Bundle Adjustment.
        """
        print("Performing Bundle Adjustment...\n")
        # print(f"Cset : \n{Cset_}")
        # print(f"Rset : \n{Rset_}")
        Cset_, Rset_, X_complete = BundleAdjustment(X_complete, X_f, features_u, features_v, inlier_feature_flag, Rset_, Cset_, K, i)


    feature_idx = np.where(X_f[:, 0])
    X = X_complete[feature_idx]
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    
    # 2D plotting
    fig = plt.figure(figsize = (10, 10))
    plt.xlim(-250,  250)
    plt.ylim(-100,  500)
    plt.scatter(x, z, marker='.',linewidths=0.5, color = 'blue')
    for i in range(0, len(Cset_)):
        R1 = Rotation.from_matrix(Rset_[i]).as_matrix()
        R1 = np.rad2deg(R1)
        plt.plot(Cset_[i][0],Cset_[i][2], marker=(3, 0, int(R1[1])), markersize=15, linestyle='None')
        
    plt.savefig(f'2D.png')
    plt.show()
    




if __name__ == '__main__':
    main()
    

