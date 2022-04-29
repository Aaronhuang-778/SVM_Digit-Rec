# 代码创作者： Aaron（黄炜）
# 联系方式：aaronwei@buaa.edu.cn
# 开发时间： 2022/4/13 14:30
from PIL import Image
import os
import sys
import numpy as np
import time
from sklearn import svm
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def get_img(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png")]


def get_img_name_str(imgPath):
    return imgPath.split(os.path.sep)[-1]

def img2vector(imgFile):
    # print("in img2vector func--para:{}".format(imgFile))
    img = Image.open(imgFile).convert('L')
    img_arr = np.array(img, 'i')  # 20px * 20px 灰度图像
    img_normalization = np.round(img_arr / 255)  # 对灰度值进行归一化
    img_arr2 = np.reshape(img_normalization, (1, -1))  # 1 * 400 矩阵
    return img_arr2


def read_and_convert(imgFileList):
    dataNum = len(imgFileList)                         # 所有图片
    dataLabel = np.zeros(dataNum,dtype=np.uint8)       # 存放类标签
    dataMat = np.zeros((dataNum, 784))                 # dataNum * 400 的矩阵(一行为一张图的数据)
    for i in range(dataNum):
        img_path = imgFileList[i]
        dataLabel[i] = img_path.split("/")[-1][0]      # 得到类标签(数字)
        dataMat[i, :] = img2vector(img_path)
    return dataMat, dataLabel

# 读取训练数据
def read_all_data(train_data_path):
    img_list = get_img(train_data_path)
    dataMat, dataLabel = read_and_convert(img_list)
    return dataMat, dataLabel

# create model
def create_svm(dataMat, dataLabel, path, decision='ovr'):
    clf = svm.SVC( C=1.0, kernel='rbf', decision_function_shape=decision)
    rf = clf.fit(dataMat, dataLabel)
    joblib.dump(rf, path)
    return clf



train_path = 'strong_train/'
dataMat, dataLabel = read_all_data(train_path)
print("data read done!")
start = time.time()
save_path = 'svm_v3.model'
create_svm(dataMat, dataLabel, save_path, decision='ovr')
end = time.time()
print("Training time {:.4f}".format(end - start))
