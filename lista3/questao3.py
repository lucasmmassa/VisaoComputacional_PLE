import numpy as np
import cv2
import os

height = 9
width = 6

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cap = cv2.VideoCapture('q3_original_video.avi') 
codec = cv2.VideoWriter_fourcc("X", "V", "I", "D")
frame_rate = 30
resolution = (640, 480)
writer = cv2.VideoWriter('q3_output_video.avi', codec, frame_rate, resolution)

if (cap.isOpened()== False):  
    print("Error opening video file") 

count = 0
   
while(cap.isOpened()):   # gerando video com marcação dos vertices do tabuleiro e salvando imagens resultantes sem distorcao
      
    ret, frame = cap.read() 
    if ret == True: 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        objp = np.zeros((height*width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

        objpoints = [] 
        imgpoints = [] 

        found, corners = cv2.findChessboardCorners(gray, (width, height), flags=cv2.CALIB_CB_FILTER_QUADS)

        if found == True:

            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
            imgpoints.append(corners2)

            frame = cv2.drawChessboardCorners(frame, (7,7), corners2,ret)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

            h,  w = frame.shape[:2]
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

            dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]

            cv2.imwrite(os.path.join('frames_sem_distorcao',str(count)+'.jpg'),dst) ## salvando o frame sem distorção
            count += 1

        cv2.imshow('Frame', frame) 
        frame = cv2.resize(frame, (640, 480))
        writer.write(frame)
   
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
   
    else:  
        break

cap.release()

writer.release()

cv2.destroyAllWindows()