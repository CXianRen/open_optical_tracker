import cv2
import numpy as np
import matplotlib.pyplot as plt



# img = cv2.imread("./img/plane/tran_test.jpg")

# rows,cols,ch = img.shape

# pts1 = np.float32([[305,208],[959,202],[73,671],[1162,664]])
# pts2 = np.float32([[0,0],[1000,0],[0,1000],[1000,1000]])

# # https://blog.csdn.net/flyyufenfei/article/details/78436675
# M = cv2.getPerspectiveTransform(pts1,pts2)

# dst = cv2.warpPerspective(img,M,(1000,1000))

# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()


class tracker():
    #左上点，右上点，左下点，右下点
    #图上标定点所在像素 x,y
    img4p_list=np.float32([[0,0],[1,0],[0,1],[1,1]])
    #标记区域的比例 宽：高
    lw_rate=(1,1)
    #缩放系数 即决定地图宽的像素大小 1000 表示宽1000像素
    scale=1000
    #比例地图的四个点位置
    #宽：高
    map4p_list=pts2 = np.float32([[0,0],[1,0],[0,1*lw[1]/lw[0]],[1,1*lw[1]/lw[0]]])*scale

    #变换矩阵M 3*3 
    M=np.float32(np.arange(9)).reshape(3,3)

    #TODO 跟踪信息,用于加快搜索

    #
    #硬件信息
    camera_id=0
    #畸变修正信息
    mtx=None
    dist=None 

    def __init__(self):
        #TODO 初始化4个坐标点/从文件读入
        pass
    
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
        pass

    #定位
    def get_relate_location(self,img):
        # 检查颜色通道
        # 图像矫正
        # 获取ROI
        # 获取点的坐标
        # 计算变换后的坐标点
        # 计算全局位置 ？
        pass
    
    #地图标定
    def set_img4p(self):
        pass
    
    #载入标定好的数据
    def load_img4p(self,path):
        pass

    

           
            

# 框架流程
# start-> 在4个camera里找到初始的 ->记录当前的
# 越界切换 在某个范围之外就进行切换
# 疯狂获取坐标
# 绘制轨迹