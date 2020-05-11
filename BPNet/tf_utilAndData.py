# Author:yifan
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import matplotlib.image as mpimg # mpimg 用于读取图片
import math

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
    test_set_y_orig=np.array([test_set_y_orig])
    train_set_y_orig=np.array([train_set_y_orig])
    classes = np.array([0 ,1, 2 ,3, 4, 5])   #手势分类
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes





def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    x = tf.placeholder("float", [12288, None])
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})
    return prediction

#辅助def predict(X, parameters)
def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3
    return Z3