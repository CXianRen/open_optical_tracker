import cv2
import numpy as np
import glob


camera="logi270"
cal_path="./test_img/"+camera+"/"
img_path="./test_img/"+camera+"/"

# TODO THERE ARE SOME BUGS IN OPENING CAMERA IN 1280*720
is_cap=0
camera_id=0

cap_h=1280
cap_w=720

test_video="test.mp4"

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
# 4*6：x y z
objp = np.zeros((S[1]*S[0],3), np.float32)
objp[:,:2] = np.mgrid[0:S[0], 0:S[1]].T.reshape(-1,2)

axis = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                   [0.5,0.5,-2],[0.5,0.5,-2],[1,1,-0],[1,0,-0] ])

try:
    if(is_cap):
        cap=cv2.VideoCapture(camera_id)
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.FOURCC('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,cap_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cap_h)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap=cv2.VideoCapture(img_path+test_video)
except Exception as e:
    print(e)
    exit(0)

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
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        # objp 实际坐标系
        # corners 像素坐标系
        # mtx dist 畸变矫正参数
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img,corners,imgpts)
        pass
    cv2.imshow('img',img)
    k=cv2.waitKey(10) & 0XFF
    if(k=='s'):
        break


cv2.destroyAllWindows()