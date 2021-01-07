import numpy as np
import cv2
import glob
import PIL.ExifTags
import PIL.Image
from tqdm import tqdm
import os

###################################################################
# CALIBRAÇÃO DA CAMERA A PARTIR DE IMAGENS DE TABULEIRO DE XADREZ #
###################################################################

chessboard_size = (6,4)

obj_points = []
img_points = []

objp = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

calibration_paths = glob.glob(os.path.join('calibration_imgs','*'))

count = 0

for image_path in tqdm(calibration_paths):

    image = cv2.imread(image_path)
    image = cv2.resize(image,(int(image.shape[1]/4),int(image.shape[0]/4)))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Analisando imagem. Formato:",image.shape)
    print("Procurando tabuleiro...")
    ret,corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)

    if ret == True:

        count += 1

        print("Tabuleiro detectado.")
        print(image_path)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray_image, corners, (5,5), (-1,-1), criteria)
        obj_points.append(objp)
        img_points.append(corners)

print(count,'tabuleiros identificados.')

print('Calibrando...')

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,gray_image.shape[::-1], None, None)

np.save(os.path.join("camera_params","ret.npy"), ret)
np.save(os.path.join("camera_params","K.npy"), K)
np.save(os.path.join("camera_params","dist.npy"), dist)
np.save(os.path.join("camera_params","rvecs.npy"), rvecs)
np.save(os.path.join("camera_params","tvecs.npy"), tvecs)

exif_img = PIL.Image.open(calibration_paths[0])
exif_data = {
 PIL.ExifTags.TAGS[k]:v
 for k, v in exif_img._getexif().items()
 if k in PIL.ExifTags.TAGS}
 
focal_length = exif_data['FocalLength']

print('Distância focal:',focal_length)

np.save(os.path.join("camera_params","FocalLength.npy"), focal_length)

print('Terminou.')