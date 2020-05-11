# Author:yifan
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import matplotlib.image as mpimg # mpimg 用于读取图片
import math
from tf_utilAndData import load_dataset

#各文件夹名称提取，返回如['data/test/0'，'data/test/1'....]的数组
def load_dataset():
    pathtest = 'data/test/'
    pathtrain = 'data/train/'
    # dirs = os.listdir('data/test')   #这个代码只用于读取文件夹的名称，不用了，直接用0-5的数字
    pathtestX,pathtrainX = [],[]
    for i in range(6):
        temptest = pathtest+str(i)
        temptrain = pathtrain + str(i)
        pathtestX.append(temptest)
        pathtrainX.append(temptrain)
    test_set_x_orig,train_set_x_orig,test_set_y_orig,train_set_y_orig = [],[],[],[]
    for i in range(6):
        testdirs = os.listdir(pathtestX[i])# 读取文件夹的名称,返回：['0.jpg', '1.jpg', '102.jpg', '106.jpg', '18.jpg', '2.jpg']
        traindirs = os.listdir(pathtrainX[i])

        for j in range(len(testdirs)):
            # 读取图片test
            pathtest = pathtestX[i]+'/'+testdirs[j]   #具体图片的路径
            test_set_x_orig.append(mpimg.imread(pathtest))  # 读取和代码处于同一目录下的 jpg
            test_set_y_orig.append(i)
        for j in range(len(traindirs)):
            # 读取图片test
            pathtrain = pathtrainX[i]+'/'+traindirs[j]   #具体图片的路径
            train_set_x_orig.append(mpimg.imread(pathtrain))  # 读取和代码处于同一目录下的 jpg
            train_set_y_orig.append(i)
    test_set_x_orig=np.array(test_set_x_orig)
    train_set_x_orig=np.array(train_set_x_orig)
    # test_set_y_orig=np.array([test_set_y_orig])
    # train_set_y_orig=np.array([train_set_y_orig])
    # classes = np.array([0 ,1, 2 ,3, 4, 5])   #手势分类
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


