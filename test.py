# 代码创作者： Aaron（黄炜）
# 联系方式：aaronwei@buaa.edu.cn
# 开发时间： 2022/4/13 15:38
import time
import os
import joblib
from PIL import Image
import numpy as np


def get_img(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".png")]

def img2vector(imgFile):
    img = Image.open(imgFile).convert('L')
    img_arr = np.array(img, 'i')                       # 28px * 28px 灰度图像
    img_normalization = np.round(img_arr / 255)        # 归一化
    img_arr2 = np.reshape(img_normalization, (1, -1))  #  矩阵
    return img_arr2

def svm_test(test_data_path,model_path):
    clf = joblib.load(model_path)                     # 加载模型
    img_list = get_img(test_data_path)
    accuracy_rate = 0
    error_count = 0
    for i in range(len(img_list)):
        img_path = img_list[i]
        dataLabel = img_path.split("/")[-1][0]
        dataMat = img2vector(img_path)
        preResult = clf.predict(dataMat)[0]
        print("num :"+str(i+1),"dataLabel :" ,dataLabel, " predict :",preResult)
        if str(preResult) != dataLabel:
            error_count+=1

        accuracy_rate = (i + 1 - error_count) / (i + 1) * 100
        print("The accuracy is: ",accuracy_rate, "%")
    print("The final accuracy is: ", accuracy_rate, "%")

if __name__ == '__main__':
    test_path = "test_image/"
    model_path = 'svm_v4.model'
    svm_test(test_path, model_path)
