import numpy as np
import cv2
import glob
import PIL.ExifTags
import PIL.Image
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
import open3d as o3d

def generateDisparityMap(left,right):
    # RECONSTRUÇÃO 3D

    ret = np.load(os.path.join("camera_params","ret.npy"))
    K = np.load(os.path.join("camera_params","K.npy"))
    dist = np.load(os.path.join("camera_params","dist.npy"))

    img_1 = cv2.imread(left)
    img_2 = cv2.imread(right)
    img_1 = cv2.resize(img_1,(int(img_1.shape[1]/4),int(img_1.shape[0]/4)))
    img_2 = cv2.resize(img_2,(int(img_2.shape[1]/4),int(img_2.shape[0]/4)))

    h,w = img_2.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

    img_1_undistorted = cv2.undistort(img_1, K, dist, None, K)
    img_2_undistorted = cv2.undistort(img_2, K, dist, None, K)

    win_size = 2
    min_disp = 0
    max_disp = 16*8 #min_disp * 9
    num_disp = max_disp - min_disp # Needs to be divisible by 16
    #Create Block matching object. 
    stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
    numDisparities = num_disp,
    blockSize = 3,
    uniquenessRatio = 5,
    speckleWindowSize = 5,
    speckleRange = 1,
    P1 = 8*3*win_size**2,#8*3*win_size**2,
    P2 =32*3*win_size**2) #32*3*win_size**2)
    #Compute disparity map
    print ("\nComputing the disparity  map...")
    disparity_map = stereo.compute(img_1_undistorted, img_2_undistorted)
    plt.imsave('disparity_map.jpg',disparity_map)