import cv2
import numpy as np
import matplotlib.pyplot as plt


# # https://blog.csdn.net/flyyufenfei/article/details/78436675

class tracker():
    def __init__(self,debug=True,lw_rate=(1,1),scale=1000):
        #TODO 初始化4个坐标点/从文件读入
        self.debug=True

        #左上点，右上点，左下点，右下点
        #图上标定点所在像素 x,y
        self.img4p_list=np.float32([[0,0],[1,0],[0,1],[1,1]])
        #标记区域的比例 宽：高
        self.lw_rate=lw_rate
        #缩放系数 即决定地图宽的像素大小 1000 表示宽1000像素
        self.scale=scale
        #比例地图的四个点位置
        #宽：高
        self.map4p_list= np.float32([[0,0],[1,0],[0,1*lw_rate[1]/lw_rate[0]],[1,1*lw_rate[1]/lw_rate[0]]])*scale

        #变换矩阵M 3*3 
        self.M=np.float32(np.arange(9)).reshape(3,3)

        #TODO 跟踪信息,用于加快搜索

        #
        #硬件信息
        self.camera_id=0
        self.w=1280
        self.h=720
        #畸变修正信息
        self.mtx=None
        self.dist=None 

        #debug
        self.debug=True
    
    #进行相机矫正
    def cam_cali(self):
        # TODO 
        pass
    
    #载入相机矫正参数
    def load_cali(self,path):
        with np.load(path) as X:
            self.mtx, self.dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

    #计算投影变换的矩阵M
    def get_m(self):
        #TODO 计算一次变换矩阵M
        self.M = cv2.getPerspectiveTransform(self.img4p_list,self.map4p_list)

    #定位
    def get_relate_location(self,img):
        # 检查颜色通道
        # 图像矫正
        # 获取ROI
        # 获取点的坐标
        # 计算变换后的坐标点
        # 计算全局位置 ？
        a=int(self.lw_rate[1]/self.lw_rate[0]*self.scale)
        dst = cv2.warpPerspective(img,self.M,(1*self.scale,int(self.lw_rate[1]/self.lw_rate[0]*self.scale)))
        if(self.debug):
            # plt.subplot(121),plt.imshow(img),plt.title('Input')
            # plt.subplot(122),plt.imshow(dst),plt.title('Output')
            # plt.show()
            cv2.imshow("res",cv2.resize(dst,(int(dst.shape[0]/3),int(dst.shape[1]/3))))
            cv2.waitKey(30)

    #地图标定
    def set_img4p(self):
        pass
    
    #载入标定好的数据
    def load_img4p(self,path):
        pass

    
if __name__ == "__main__":
    t_1=tracker(lw_rate=(1,1))
    t_1.scale=1100
    t_1.img4p_list=np.float32([[289,30],[953,26],[106,569],[1103,567]])
    t_1.get_m()
    cap=cv2.VideoCapture("./img/plane/test_video.mp4")
    while 1:
        ret,f =cap.read()
        if(ret):
            t_1.get_relate_location(f)
        else:
            break


# 框架流程
# start-> 在4个camera里找到初始的 ->记录当前的
# 越界切换 在某个范围之外就进行切换
# 疯狂获取坐标
# 绘制轨迹