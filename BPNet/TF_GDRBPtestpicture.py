# Author:yifan
import math
from tensorflow.python.framework import ops
from tf_utilAndData import load_dataset,predict,predict,forward_propagation_for_predict
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import matplotlib.image as mpimg # mpimg 用于读取图片
from sklearn.metrics import confusion_matrix



#one-hot编码和数据导入，用于计算准确率
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset() #取数据
Y_test = convert_to_one_hot(Y_test_orig, 6)   #返回的数据(6, 120)的one-hot编码
Y_train = convert_to_one_hot(Y_train_orig, 6)
X_train_flaten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flaten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
X_train = X_train_flaten/255
X_test = X_test_flaten/255



#线性参数
def initializer_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable('W1', [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1', [25,1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    W2 = tf.get_variable('W2', [12,25], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2', [12,1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    W3 = tf.get_variable('W3', [6,12], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3', [6,1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
    parameters = {'W1':W1, 'W2':W2, 'W3':W3, 'b1':b1, 'b2':b2, 'b3':b3}
    # print(W1[0])
    # print('111111111')
    return parameters

#设置占位符，后面使用再赋值
def creat_placeholder(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x,None) ,name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y,None) ,name='Y')
    return X ,Y

# parameters = initializer_parameters()   #parameters['W1']为<tf.Variable 'W1:0' shape=(25, 12288) dtype=float32_ref>

#载入图片数据
pathtest1 = 'data/test/0/0.jpg'
pathtest1 = mpimg.imread(pathtest1)
test1_set_x = np.array([pathtest1])  # 注意需要多个【】，因为要求反馈数据类型为(1, 64, 64, 3)
X_test_flaten1 = test1_set_x.reshape(test1_set_x.shape[0], -1).T

with tf.Session() as sess:
    X, Y = creat_placeholder(12288, 6)
    parameters = initializer_parameters()
    sess.run(tf.global_variables_initializer())
    # 调用存储的模型
    sess = tf.InteractiveSession()
    saver = tf.train.Saver(parameters)
    saver.restore(sess, "model/./cnn_model.ckpt")  # 调出上次训练好的模型
    parameters = sess.run(parameters)
    print(X_test_flaten1.shape)
    X_test1 = X_test_flaten1 / 255

    correct_prediction1 = predict(X_test1, parameters)
    print(correct_prediction1)
    print('test fininsh')

    # 混淆矩阵：
    Y_test_orig = Y_test_orig[0]  #原来的为【120，1】
    X_test = X_test_flaten / 255
    ytest_p = predict(X_test, parameters)
    ytest_p = np.array(ytest_p)
    print(ytest_p)
    print(Y_test_orig)
    Conf_Mat = confusion_matrix(Y_test_orig, ytest_p)
    print("测试集的混淆矩阵如下：")
    print("行为真实值，列为预测值")
    print(Conf_Mat)
    ytrain_p = predict(X_train, parameters)
    ytrain_p = np.array(ytrain_p)
    Acctrain = np.mean(ytrain_p == Y_train_orig)
    print("训练集合的准确率：")
    print(Acctrain)
    Acctest = np.mean(ytest_p == Y_test_orig)
    print("测试集合的准确率：")
    print(Acctest)

    # print("Parameters have benn trained!")
    # correct_prediction = tf.equal(tf.argmax(X_test), tf.argmax(Y))
    # print(correct_prediction)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # # print(accuracy)
    # print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
    # print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))