import numpy as np
import cv2
import glob

class camera_cali():

	def __init__(self,chessboard_size=(6,4)):
		self.S=chessboard_size
		# termination criteria
		self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		self.S=(6,4)
		self.objp = np.zeros((S[1]*S[0],3), np.float32)
		self.objp[:,:2] = np.mgrid[0:S[0], 0:S[1]].T.reshape(-1,2)

		# Arrays to store object points and image points from all the images.
		self.objpoints = [] # 3d points in real world space
		self.imgpoints = [] # 2d points in image plane.

	def cali_video(self,path):
		pass

	def cali_img(self,path,debug=True):
		images = glob.glob(path+"*.jpg")
		for idx, fname in enumerate(images):
			img = cv2.imread(fname)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			# Find the chessboard corners
			ret, corners = cv2.findChessboardCorners(gray,self.S, None)
			if ret == True:
				self.objpoints.append(self.objp)
				cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
				self.imgpoints.append(corners)
				# Draw and display the corners
				cv2.drawChessboardCorners(img, self.S, corners,ret)
				if(debug):
					cv2.imshow('img',img)
					cv2.waitKey(20)
		cv2.destroyAllWindows()
		ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1],None,None)
		if(debug):
			print("mtx:",self.mtx)
			print("dist:",self.dist)

	def cali_cam(self,camera_id=0):
		pass

	def save_cali(self,path):
		try:
			np.savez(path,mtx=self.mtx, dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs)
		except Exception as e:
			print(e)
			print("you should run obj.cali_x() before you save it")
	
	def test(self,img_path):
		img = cv2.imread(img_path)
		h,  w = img.shape[0:2]
		newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),1,(w,h))

		# undistort
		dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)

		# crop the image
		x,y,w,h = roi
		dst = dst[y:y+h, x:x+w]
		cv2.imwrite('calibresult.png',dst)


		mean_error = 0
		for i in range(len(self.objpoints)):
			imgpoints2, _ = cv2.projectPoints(self.objpoints[i],self.rvecs[i],self.tvecs[i], self.mtx, self.dist)
			error = cv2.norm(self.imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
			mean_error +=error

		print ("total error: ", mean_error/len(self.objpoints))


if __name__ == "__main__":
	#sizeof chessboard
	S=(6,4)
	camera="logi270"
	chessboard_path="./img/"+camera+"/"
	test_img_path="./test_img/"+camera+"/"
	cal_result_save_path="./test_img/"+camera+"/"
	run_test=False


	cc=camera_cali()
	cc.cali_img(chessboard_path)
	cc.save_cali(cal_result_save_path+"logi270.npz")

