import cv2
import numpy as np
import glob

cal_path="./test_img/servo_camera/"
img_path="./test_img/servo_camera/"
S=(6,4)

# Load previously saved data
with np.load(cal_path+'save_cal.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    
        # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    
        # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    
        # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((S[1]*S[0],3), np.float32)
objp[:,:2] = np.mgrid[0:S[0], 0:S[1]].T.reshape(-1,2)

axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

cap=cv2.VideoCapture(img_path+"WIN_20190812_18_00_44_Pro.mp4")


# for fname in glob.glob(img_path+'*.jpg'):
    # img = cv2.imread(fname)
while 1:
    ret,img =cap.read()

    if ret:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, S,None)
    else:
        print("no more frame")
        break

    if ret:
        # corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img,corners,imgpts)
    cv2.imshow('img',img)
    cv2.waitKey(30)


cv2.destroyAllWindows()