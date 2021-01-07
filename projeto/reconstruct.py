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


def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

# RECONSTRUÇÃO 3D

ret = np.load(os.path.join("camera_params","ret.npy"))
K = np.load(os.path.join("camera_params","K.npy"))
dist = np.load(os.path.join("camera_params","dist.npy"))

img_path1 = './test_imgs/5.jpg'
img_path2 = './test_imgs/6.jpg'
img_1 = cv2.imread(img_path1)
img_2 = cv2.imread(img_path2)
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

#Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
# plt.imshow(disparity_map,'gray')
# plt.show()

#Generate  point cloud. 
print ("\nGenerating the 3D map...")
focal_length = np.load(os.path.join("camera_params","FocalLength.npy"), allow_pickle=True)
#Perspective transformation matrix
#This transformation matrix is from the openCV documentation, didn't seem to work for me. 
Q = np.float32([[1,0,0,-w/2.0],
    [0,-1,0,h/2.0],
    [0,0,0,-focal_length],
    [0,0,1,0]])
#This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision. 
#Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
Q2 = np.float32([[1,0,0,0],
    [0,-1,0,0],
    [0,0,focal_length*0.05,0], #Focal length multiplication obtained experimentally. 
    [0,0,0,1]])
#Reproject points into 3D
points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
#Get color points
colors = cv2.cvtColor(img_1_undistorted, cv2.COLOR_BGR2RGB)
#Get rid of points with value 0 (i.e no depth)
mask_map = disparity_map > disparity_map.min()
#Mask colors and points. 
output_points = points_3D[mask_map]
output_colors = colors[mask_map]
#Define name for output file
output_file = 'reconstructed.ply'
#Generate point cloud 
print ("\n Creating the output file... \n")
create_output(output_points, output_colors, output_file)

cloud = o3d.io.read_point_cloud('reconstructed.ply') # Read the point cloud
o3d.visualization.draw_geometries([cloud])