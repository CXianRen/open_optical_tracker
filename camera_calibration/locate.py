# import cv2
# import numpy as np

# # https://blog.csdn.net/ColdWindHA/article/details/82080176
# if __name__ == "__main__":
#     cap=cv2.VideoCapture("./img/plane/test_red3.mp4")
#     last_f=None
#     try:
#         _,last_f=cap.read()
#         last_f = cv2.cvtColor(last_f,cv2.COLOR_BGR2GRAY)
#         h,w=last_f.shape[0:2]
#     except Exception as e:
#         exit(0)

#     while 1:
#         ret,f =cap.read()
#         if(ret):
#             f=cv2.GaussianBlur(f,(21,21),0)
#             f=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
#             h=f[:,:,0]
#             low_range = np.array([0, 80, 46])
#             high_range = np.array([10,255, 255])
#             th_1 = cv2.inRange(f, low_range, high_range)
#             low_range = np.array([160, 80, 46])
#             high_range = np.array([180,255, 255])
#             th_2 = cv2.inRange(f, low_range, high_range)
#             th=cv2.add(th_1,th_2)
#             cv2.imshow("h",h)
#             cv2.imshow("f",f)
#             # dif=cv2.absdiff(f,last_f)
#             # last_f=f
#             # t=cv2.cv2.threshold(dif,25,255,cv2.THRESH_BINARY)[1]
#             #cv2.imshow("res",cv2.resize(f,(int(w/3),int(h/3))))
#             cv2.imshow("th",th)
#             cv2.waitKey(0)
#         else:
#             break



    # f=cv2.imread("./img/plane/test_red_img.jpg")
    # f=cv2.GaussianBlur(f,(21,21),0)
    # f=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
    # h=f[:,:,0]
    # s=f[:,:,1]
    # v=f[:,:,2]
    # cv2.imshow("h",h)
    # cv2.imshow("s",s)
    # cv2.imshow("v",v)
    # cv2.waitKey(0)
    # low_range = np.array([0, 123, 100])
    # high_range = np.array([5, 255, 255])
    # th = cv2.inRange(f, low_range, high_range)
    # cv2.imshow("th",th)
    # cv2.waitKey(0)
# https://zhuanlan.zhihu.com/p/69999853

# import numpy as np
# import cv2 as cv
# import math
# cap = cv.VideoCapture('D:\JIMI\物料\视频\WIN_20180815_15_04_51_Pro.mp4')

# #角点检测参数
# feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=7, blockSize=7)

# #KLT光流参数
# lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.02))


# # 随机颜色
# color = np.random.randint(0,255,(100,3))

# # 读取第一帧
# ret, old_frame = cap.read()
# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params,useHarrisDetector=False,k=0.04)
# good_ini=p0.copy()

# def caldist(a,b,c,d):
#     return abs(a-c)+abs(b-d)

# mask = np.zeros_like(old_frame)
# # 光流跟踪
# while True:
#     ret, frame = cap.read()
#     if ret is False:
#         break
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # 计算光流
#     p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#     # 根据状态选择

#     good_new = p1[st == 1]
#     good_old = p0[st == 1]
#     #删除静止点
#     k=0
#     for i, (new0, old0) in enumerate(zip(good_new,good_old)):
#         a0,b0 = new0.ravel()
#         c0,d0 = old0.ravel()
#         dist=caldist(a0,b0,c0,d0)
#         if dist>2:
#             good_new[k]=good_new[i]
#             good_old[k]=good_old[i]
#             good_ini[k]=good_ini[i]
#             k=k+1

#     # 提取动态点
#     good_ini=good_ini[:k]
#     good_new=good_new[:k]
#     good_old=good_old[:k]  

#     # 绘制跟踪线
#     for i, (new, old) in enumerate(zip(good_new,good_old)):
#         a,b = new.ravel()
#         c,d = old.ravel()
#         mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#         frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
#     cv.imshow('frame',cv.add(frame,mask))
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         cv.imwrite("flow.jpg", cv.add(frame,mask))
#         break

#     # 更新
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1, 1, 2)

#     if good_ini.shape[0]<40:
#         p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
#         good_ini=p0.copy()

# cv.destroyAllWindows()
# cap.release()


