# 代码创作者： Aaron（黄炜）
# 联系方式：aaronwei@buaa.edu.cn
# 开发时间： 2022/4/13 16:31
import joblib
from sklearn import svm
import numpy as np
from time import time
from sklearn.metrics import accuracy_score
from struct import unpack
from sklearn.model_selection import GridSearchCV


def readimage(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img


def readlabel(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab


def main():
    train_data = readimage("dataset/train-images.idx3-ubyte")
    train_label = readlabel("dataset/train-labels.idx1-ubyte")
    test_data = readimage("dataset/t10k-images.idx3-ubyte")
    test_label = readlabel("dataset/t10k-labels.idx1-ubyte")
    svc = svm.SVC()
    parameters = {'kernel': ['rbf'], 'C': [0.8, 0.9, 1, 1.1, 1.2]}
    print("Train...")
    clf = GridSearchCV(svc, parameters, n_jobs=-1)
    clf.fit(train_data, train_label)
    joblib.dump(clf, 'svm_v4.model')


if __name__ == '__main__':
    main()