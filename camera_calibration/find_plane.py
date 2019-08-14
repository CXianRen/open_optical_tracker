import cv2
import numpy as np

camera="logi270"
cal_path="./test_img/"+camera+"/"
img_path="./test_img/"+camera+"/"

img = cv2.imread('./img/plane/plane_test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(gray, 80, 250, cv2.THRESH_BINARY_INV)
cv2.imshow("draw_img1", binary )

_,contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
draw_img1 = cv2.drawContours(img.copy(),contours,-1,(255,0,255),3)
cv2.imshow("draw_img1", draw_img1)


point_color = (0, 0, 255)
thickness = 4
point_size = 1

for i in range(0,len(contours)): 
    # 获取最小外接矩形
    rect=cv2.minAreaRect(contours[i])
    box=cv2.boxPoints(rect)
    box = np.int0(box)
    for i in box:
        cv2.circle(draw_img1, (i[0],i[1]), point_size, point_color, thickness)

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

# 2*2*3：x y z
objp = np.zeros((2*2,3), np.float32)
objp[:,:2] = np.mgrid[0:2, 0:2].T.reshape(-1,2)

slen=47
axis = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                   [0.5,0.5,-2],[0.5,0.5,-2],[1,1,-0],[1,0,-0]])
print(axis)
corners= np.float32([[[516,241]],[[808,275]],[[473,521]],[[789,556]]])
        # Find the rotation and translation vectors.
        # objp 实际坐标系
        # corners 像素坐标系
        # mtx dist 畸变矫正参数
robjp=objp*slen
_, rvecs, tvecs, inliers = cv2.solvePnPRansac(robjp, corners, mtx, dist)
print("t",tvecs)
        # project 3D points to image plane
imgpts, jac = cv2.projectPoints(axis*slen, rvecs, tvecs, mtx, dist)

img = draw(img,corners,imgpts)

cv2.imshow('img',img)
k=cv2.waitKey(0) & 0XFF

cv2.destroyAllWindows()s