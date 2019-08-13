import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("./resultimg/result11.jpg")

# print(H_rows, W_cols)

# # 原图中书本的四个角点(左上、右上、左下、右下),与变换后矩阵位置
# pts1 = np.float32([[484,190,], [ 881,244], [ 148,476], [ 727,604]])
# pts2 = np.float32([[0, 0],[W_cols,0],[0, H_rows],[H_rows,W_cols],])

# # 生成透视变换矩阵；进行透视变换
# M = cv2.getPerspectiveTransform(pts1, pts2)
# dst = cv2.warpPerspective(img, M, ( 400,600))

# # plt.subplot(121), plt.imshow(img[:, :, ::-1]), plt.title('input')
# # plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')
# # plt.show()
# cv2.imshow("original_img",img)
# cv2.imshow("result",dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread('./sudoku.png')
rows,cols,ch = img.shape

pts1 = np.float32([[484,190],[877,243],[136,476],[726,600]])
pts2 = np.float32([[0,0],[500,0],[0,700],[500,700]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(500,700))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()