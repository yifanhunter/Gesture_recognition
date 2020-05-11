# Author:yifan
import tensorflow as tf
import math
import numpy as np
import tensorflow.contrib.slim.nets as nets
from TF_GDR02_Data import load_dataset
import matplotlib.image as mpimg # mpimg 用于读取图片
from sklearn.metrics import confusion_matrix
#占位符
x = tf.placeholder(tf.float32, [None, 64, 64, 3])
y_ = tf.placeholder(tf.float32, [None])

# 将label值进行onehot编码
one_hot_labels = tf.one_hot(indices=tf.cast(y_, tf.int32), depth=6)
pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=6, is_training=True)
pred = tf.reshape(pred, shape=[-1, 6])
a = tf.argmax(pred, 1)


#测试
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset() #取数据
X_train = X_train_orig/255
X_test = X_test_orig/255


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 调用存储的模型
    saver = tf.train.Saver()
    saver.restore(sess, "GR_model03/./cnn_model.ckpt")  # 调出上次训练好的模型
    # 载入图片数据
    pathtest1 = 'data/test/5/60.jpg'
    pathtest1 = mpimg.imread(pathtest1) / 255
    test1_set_x = np.array([pathtest1])  # 数据类型为( 1,64, 64, 3)
    ytest_p = sess.run(a, feed_dict={x: test1_set_x})
    print(ytest_p)
    print('test fininsh')

    #混淆矩阵：
    ytest_p = sess.run(a, feed_dict={x: X_test})
    Conf_Mat = confusion_matrix(Y_test_orig, ytest_p)
    print("测试集的混淆矩阵如下：")
    print("行为真实值，列为预测值")
    print(Conf_Mat)
    ytrain_p = sess.run(a, feed_dict={x: X_train})
    Acctrain = np.mean(ytrain_p == Y_train_orig)
    print("训练集合的准确率：")
    print(Acctrain)
    #
    Acctest = np.mean(ytest_p == Y_test_orig)
    print("测试集合的准确率：")
    print(Acctest)
    # print(ytest_p.shape)   #(120,)


