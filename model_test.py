# 代码创作者： Aaron（黄炜）
# 联系方式：aaronwei@buaa.edu.cn
# 开发时间： 2022/4/14 12:10
import joblib
from sklearn import svm
import numpy as np
from time import time
from sklearn.metrics import accuracy_score
from struct import unpack
from sklearn.model_selection import GridSearchCV
import os
from PIL import Image


def readimage(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img

def get_img(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png")]


def readlabel(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab

def img2vector(imgFile):
    img = Image.open(imgFile).convert('L')
    img_arr = np.array(img, 'i')                       # 28px * 28px 灰度图像
    return img_arr.reshape(1, 784)

def svm_test1(model_path, test_path):
    clf = joblib.load(model_path)  # 加载模型
    img_list = get_img(test_path)
    accuracy_rate = 0
    error_count = 0
    for i in range(len(img_list)):
        img_path = img_list[i]
        dataLabel = img_path.split("/")[-1][0]
        dataMat = img2vector(img_path)
        preResult = clf.predict(dataMat)[0]
        print("NO. ", i + 1, " The input number is :", dataLabel, " your prediction is :", preResult)

        if str(preResult) != dataLabel:
            error_count += 1

        accuracy_rate = (i + 1 - error_count) / (i + 1) * 100
        print("The accuracy is: ", accuracy_rate, "%")
    print("The final accuracy is: ", accuracy_rate, "%")

def svm_test(model_path, test_path, label_path):
    test_data = readimage(test_path)
    test_label = readlabel(label_path)
    clf = joblib.load(model_path)
    prediction = clf.predict(test_data)

    i = 0
    while i < len(test_label):
        if prediction[i] == test_label[i]:
            print("NO. ", i+1, " The input number is :",  test_label[i], " your prediction is :", prediction[i])
        i += 1
    print("accuracy: ", accuracy_score(prediction, test_label) *100, "%")

if __name__ == '__main__':
    model_path = "svm_v4.model"
    test_ubyte_path = "dataset/t10k-images.idx3-ubyte"
    label_path = "dataset/t10k-labels.idx1-ubyte"
    test_img_path = "test_image/"


    #svm_test(model_path, test_ubyte_path, label_path)
    svm_test1(model_path, test_img_path)