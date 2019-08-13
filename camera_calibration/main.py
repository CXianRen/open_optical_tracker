import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


#sizeof chessboard
S=(6,4)
camera="logi270"
chessboard_path="./img/"+camera+"/"
test_img_path="./test_img/"+camera+"/"
cal_result_save_path="./test_img/"+camera+"/"

run_test=False

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((S[1]*S[0],3), np.float32)
objp[:,:2] = np.mgrid[0:S[0], 0:S[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(chessboard_path+"*.jpg")
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, S, None)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, S, corners,ret)
        # cv2.imshow('img',img)
        # cv2.waitKey(20)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


if(run_test):
	img = cv2.imread(test_img_path+'test.jpg')
	h,  w = img.shape[0:2]
	newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))


	# undistort
	dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

	# crop the image
	x,y,w,h = roi
	dst = dst[y:y+h, x:x+w]
	cv2.imwrite('calibresult.png',dst)


	mean_error = 0
	for i in range(len(objpoints)):
		imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
		error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
		mean_error +=error

	print ("total error: ", mean_error/len(objpoints))

np.savez(cal_result_save_path+"save_cal.npz",mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
# import pickle

# # Test undistortion on an image
# img = cv2.imread('./tftest.jpg')

# img_size = (img.shape[1], img.shape[0])

# # Do camera calibration given object points and image points
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,(1280,720),None,None)
# print("mtx:")
# print(mtx)
# print("dist:")
# print(dist)

# dst = cv2.undistort(img, mtx, dist, None, mtx)
# cv2.imwrite('resultimg/result11.jpg',dst)

# # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
# dist_pickle = {}
# dist_pickle["mtx"] = mtx
# dist_pickle["dist"] = dist
# pickle.dump( dist_pickle, open( "./wide_dist_pickle.p", "wb" ) )
# #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# # Visualize undistortion

# # delt=dst-img
# #cv2.imshow('result', delt)



