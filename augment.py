# 代码创作者： Aaron（黄炜）
# 联系方式：aaronwei@buaa.edu.cn
# 开发时间： 2022/4/13 15:53
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pdb
from autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, SubPolicy
from autoaugment import ImageNetPolicy
import  os

policy = SVHNPolicy()
imgs_dir = "train_image/"
image_names = os.listdir(imgs_dir)
os.makedirs("strong_train/")
i = 0
for f in image_names:
    img0 = Image.open("train_image/" + f)
    img0.save("strong_train/" + f + str(i) + ".png")
    img = img0.convert('RGB')
    i += 1
    for _ in range(5):
        img1 = policy(img).convert('L')
        print(i)
        img1.save("strong_train/" + f + str(i) + ".png")
        i += 1
    print("Has done: " + f)


