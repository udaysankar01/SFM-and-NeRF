import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def readImages(data_path):
    """
    Read a number of images from the given path of folder containing the images.

    Parameters
    ----------
    data_path : string 
        path of folder containing the images

    Results
    -------
    images : array-like
        array of images withing the given folder
    """

    print(f'Reading images from "{data_path}"')
    images = []

    for file in os.listdir(data_path):
        
        # check if file ends with '.png'
        if file.endswith('.png'):
            image = cv2.imread(f'{data_path}/{file}')
            images.append(image)
    
    images = np.array(images)
    return images


def getMatchingFeatures(data_path, images):
    """
    Gets the matching features between images from the 'mathcing*.txt' files.

    Parameters
    ----------
    data_path : string 
        path of folder containing the images
    images : array-like
        array of images withing the given folder

    Results
    -------
    features_u : array-like
        array of features
    features_v : array-like
        array of features
    flag_feature : array-like
        array containing 1sand 0s repersenting whether match is there or not
    """

    features_u = []
    features_v = []
    flag_feature = []
    for image_number in range(1, images.shape[0]):
        
        matching_file_path = f'{data_path}/matching{image_number}.txt'
        f = open(matching_file_path, 'r')

        for i, line in enumerate(f):
            
            # to get nFeatures
            if i == 0:
                nFeatures = int(re.findall(r'\d+', line)[0])
            
            else:
                urow = np.zeros(images.shape[0])
                vrow = np.zeros(images.shape[0])
                flag = np.zeros(images.shape[0], dtype=int)

                line_items = np.array([float(num) for num in line.split()])
                
                n_matches = line_items[0]

                r, g, b = line_items[1: 4]

                curr_u = line_items[4]
                curr_v = line_items[5]
                urow[image_number - 1] = curr_u
                vrow[image_number - 1] = curr_v
                flag[image_number - 1] = 1

                match_idx = 1
                while n_matches > 1:
                    curr_img = int(line_items[5 + match_idx])
                    curr_img_u = line_items[6 + match_idx]
                    curr_img_v = line_items[7 + match_idx]

                    match_idx += 3

                    urow[curr_img - 1] = curr_img_u
                    vrow[curr_img - 1] = curr_img_v
                    flag[curr_img - 1] = 1
                    n_matches -= 1
                
                features_u.append(np.transpose(urow))
                features_v.append(np.transpose(vrow))
                flag_feature.append(np.transpose(flag))

    return np.array(features_u).reshape(-1, images.shape[0]), np.array(features_v).reshape(-1, images.shape[0]), np.array(flag_feature).reshape(-1, images.shape[0])


def sameSizeImages(images):
    """
    Resizes all input images to same size for visualization.

    Parameters
    ----------
    images : array-like
        array containing all the input images

    Results
    -------
    resized_images : array-like
        array containing all the resized images
    """
    
    imgs = images.copy()
    image_sizes = np.array([img.shape for img in imgs])
    max_shape = np.max(image_sizes, axis=0)
    
    resized_images = []

    for i, img in enumerate(imgs):
        resized_img = np.zeros(max_shape, np.uint8)
        resized_img[0: image_sizes[i, 0], 0: image_sizes[i, 1], 0: image_sizes[i, 2]] = img
        resized_images.append(resized_img)
    
    resized_images = np.array(resized_images)
    return resized_images


def displayMatches(image1, image2, pts1, pts2, file_path):
    """
    Displays the matching between two images.

    Parameters
    ----------
    image1 : array-like
        array representing first image
    image2 : array-like
        array representing second image
    pts1 : array-like
        points for matching in first image
    pts2 : array-like
        points for matching in second image
    path : string
        path of debug folder for storing results
    """
    img1, img2 = sameSizeImages([image1, image2])
    points1 = pts1.copy()
    points2 = pts2.copy()

    concat = np.concatenate((img1, img2), axis = 1)

    features_1u = points1[:, 0].astype(int)
    features_1v = points1[:, 1].astype(int)
    features_2u = points2[:, 0].astype(int) + img1.shape[1]
    features_2v = points2[:, 1].astype(int)

    for i in range(len(features_1u)):
        cv2.line(concat, (features_1u[i], features_1v[i]), (features_2u[i], features_2v[i]), color=(255,0,0), thickness=1)
        cv2.circle(concat, (features_1u[i], features_1v[i]), 2, color=(0,0,255), thickness=1)
        cv2.circle(concat, (features_2u[i], features_2v[i]), 2, color=(0,255,255), thickness=1)
    cv2.imwrite(file_path, concat)

def displayMatchesAfterRANSAC(image1, image2, inlier_pts1, inlier_pts2, outlier_pts1, outlier_pts2, file_path):
    """
    Displays the matching between two images.

    Parameters
    ----------
    image1 : array-like
        array representing first image
    image2 : array-like
        array representing second image
    inlier_pts1 : array-like
        inlier points for matching in first image
    inlier_pts2 : array-like
        inlier points for matching in second image
    outlier_pts1 : array-like
        outlier points for matching in first image
    outlier_pts2 : array-like
        outlier points for matching in second image
    path : string
        path of debug folder for storing results
    """
    img1, img2 = sameSizeImages([image1, image2])
    inlier_points1 = inlier_pts1.copy()
    inlier_points2 = inlier_pts2.copy()
    outlier_points1 = outlier_pts1.copy()
    outlier_points2 = outlier_pts2.copy()

    concat = np.concatenate((img1, img2), axis = 1)

    inlier_features_1u = inlier_points1[:, 0].astype(int)
    inlier_features_1v = inlier_points1[:, 1].astype(int)
    inlier_features_2u = inlier_points2[:, 0].astype(int) + img1.shape[1]
    inlier_features_2v = inlier_points2[:, 1].astype(int)

    outlier_features_1u = outlier_points1[:, 0].astype(int)
    outlier_features_1v = outlier_points1[:, 1].astype(int)
    outlier_features_2u = outlier_points2[:, 0].astype(int) + img1.shape[1]
    outlier_features_2v = outlier_points2[:, 1].astype(int)

    # for i in range(len(outlier_features_1u)):
    #     cv2.line(concat, (outlier_features_1u[i], outlier_features_1v[i]), (outlier_features_2u[i], outlier_features_2v[i]), color=(0,0,255), thickness=1)
    #     cv2.circle(concat, (outlier_features_1u[i], outlier_features_1v[i]), 2, color=(255,0,255), thickness=1)
    #     cv2.circle(concat, (outlier_features_2u[i], outlier_features_2v[i]), 2, color=(0,255,255), thickness=1)

    for i in range(len(inlier_features_1u)):
        cv2.line(concat, (inlier_features_1u[i], inlier_features_1v[i]), (inlier_features_2u[i], inlier_features_2v[i]), color=(0,255,0), thickness=1)
        cv2.circle(concat, (inlier_features_1u[i], inlier_features_1v[i]), 2, color=(255,0,255), thickness=1)
        cv2.circle(concat, (inlier_features_2u[i], inlier_features_2v[i]), 2, color=(0,255,255), thickness=1)

    cv2.imwrite(file_path, concat)


def visualizeEpipoles(image1, image2, points1, points2, F, filename1, filename2):                  ########################################################
    

    e1 = np.linalg.svd(F)[2][-1]
    e2 = np.linalg.svd(F.T)[2][-1]

    e1 = e1 / e1[2]
    e2 = e2 / e2[2]

    img1 = image1.copy()
    for i in range(len(points1)):
        cv2.line(img1, points1[i].astype(int), tuple(e1[:2].astype(int)), (255, 0, 0), 1)

    cv2.imwrite(filename1, img1)

    img2 = image2.copy()
    for i in range(len(points2)):
        cv2.line(img2, points2[i].astype(int), tuple(e2[:2].astype(int)), (255, 0, 0), 1)

    cv2.imwrite(filename2, img2)



def ReprojErrorPnP(X, x, K, R, C):

    error = []

    C_ = np.reshape(C, (3, 1))        
    P = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C_))))

    for x_, X_ in zip(x, X):
        u, v = x_[0], x_[1]
        P_1T, P_2T, P_3T = P[0].reshape(1, -1), P[1].reshape(1, -1), P[2].reshape(1, -1)
        X_ = X_.reshape(1, -1)
        X_ = np.hstack((X_, np.ones((X_.reshape(1, -1).shape[0], 1)))).reshape(-1, 1)
        e = np.square(u - np.divide(np.dot(P_1T, X_), np.dot(P_3T, X_))) + np.square(v - np.divide(np.dot(P_2T, X_), np.dot(P_3T, X_)))
        error.append(e)
    
    meanError = np.mean(np.array(error).squeeze())

    return meanError    


def meanReprojError(X, x1, x2, R1, C1, R2, C2, K):

    error = []
    I = np.identity(3)

    C1 = np.reshape(C1, (3, 1))        
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))

    C2 = np.reshape(C2, (3, 1))        
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    P1_1T, P1_2T, P1_3T = P1[0].reshape(1, -1), P1[1].reshape(1, -1), P1[2].reshape(1, -1)
    P2_1T, P2_2T, P2_3T = P2[0].reshape(1, -1), P2[1].reshape(1, -1), P2[2].reshape(1, -1)

    for pt1, pt2, x in zip(x1, x2, X):
        u1, v1 = pt1
        u2, v2 = pt2
        error1 = np.square(u1 - np.divide(np.dot(P1_1T, x), np.dot(P1_3T, x))) + np.square(v1 - np.divide(np.dot(P1_2T, x), np.dot(P1_3T, x)))
        error2 = np.square(u2 - np.divide(np.dot(P2_1T, x), np.dot(P2_3T, x))) + np.square(v2 - np.divide(np.dot(P2_2T, x), np.dot(P2_3T, x)))
        error.append(error1 + error2)
    
    meanError = np.mean(error)

    return meanError


def visualize2DPlot(X, Cset, Rset, filename):
    
    x = X[:, 0]
    z = X[:, 2]

    fig = plt.figure(figsize=(10, 10))
    plt.xlim(-250, 250)
    plt.ylim(-100,500)
    plt.scatter(x, z, marker='.', linewidths=0.5, color='red')
    
    for i in range(len(Cset)):  
        R1 = Rotation.from_matrix(Rset[i]).as_rotvec()
        R1 = np.rad2deg(R1)
        plt.plot(Cset[i][0], Cset[i][2], marker=(3, 0, int(R1[1])), markersize=15, linestyle='None')

    plt.savefig(filename)
    plt.show()
