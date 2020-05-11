# Author:yifan

import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utilAndData import load_dataset,   predict
import matplotlib.image as mpimg # mpimg 用于读取图片

#one-hot编码
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset() #取数据
Y_test = convert_to_one_hot(Y_test_orig, 6)   #返回的数据(6, 120)的one-hot编码
Y_train = convert_to_one_hot(Y_train_orig, 6)

#X数据的处理，最终是0-1的数据，每一列是一个64*64*3的数据（表示一个图）
#返回(12288, 1080)，表示64*64*3个数据展开成一列，共1080个数据。reshape的-1表示不限制列数
X_train_flaten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flaten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
X_train = X_train_flaten/255
X_test = X_test_flaten/255


#设置占位符，后面使用再赋值
def creat_placeholder(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x,None) ,name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y,None) ,name='Y')
    return X ,Y

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
parameters = initializer_parameters()   #parameters['W1']为<tf.Variable 'W1:0' shape=(25, 12288) dtype=float32_ref>

#前向传播过程
def forward_propagation(x, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    b1 = parameters['b1']
    b2 = parameters['b2']
    b3 = parameters['b3']
    out = tf.matmul(W1, x) + b1
    out = tf.nn.relu(out)
    out = tf.matmul(W2, out) + b2
    out = tf.nn.relu(out)
    out = tf.matmul(W3, out) + b3
    return out

#定义计算算损失函数
def compute_cost(out, y):
    logits = tf.transpose(out) #转置
    labels = tf.transpose(y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]                           #(12288, 1080)
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))   # (6, 1080)
    # print("--------------")
    # print(shuffled_X.shape)
    # print(shuffled_Y.shape)
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

# #第一步：前向传播，这一步会
tf.reset_default_graph()
# with tf.Session() as sess:
#     X, Y = creat_placeholder(12288, 6)
#     parameters = initializer_parameters()
#     out = forward_propagation(X, parameters)
#     cost = compute_cost(out, Y)

# def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,nums_epoch=1000, minibatch_size=32, print_cost=True):
with tf.Session() as sess:

    X, Y = creat_placeholder(12288, 6)
    parameters = initializer_parameters()
    out = forward_propagation(X, parameters)
    cost = compute_cost(out, Y)

    sess.run(tf.global_variables_initializer())
    learning_rate=0.0001
    nums_epoch=1000   #1000比较好
    minibatch_size=32
    print_cost=True
    # tf.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape  #本次反回12288  1080
    n_y = Y_train.shape[0]
    costs = []
    print(m)
    X, Y = creat_placeholder(n_x, n_y)   #执行后的结果设置占位符,12288 6
    # parameters = initializer_parameters()
    out = forward_propagation(X, parameters)
    cost = compute_cost(out, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())   #就是 run了 所有global Variable 的 assign op，这就是初始化参数的本来面目
        for epoch in range(nums_epoch):
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)   #1080/32  表示训练minibatches的数量
            seed = seed + 1 #每次seed不一样，不同的sess号会不同
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch  #切成小块的minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})  #按照小模块进行训练
                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        # save model
        saver = tf.train.Saver(parameters)
        saver.save(sess, "model//cnn_model.ckpt")




        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate = " + str(learning_rate))
        # plt.show()
        # parameters = sess.run(parameters)
        print("Parameters have benn trained!")
        correct_prediction = tf.equal(tf.argmax(out), tf.argmax(Y))
        print(correct_prediction)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        # return parameters
# parameters = model(X_train , Y_train, X_test, Y_test)
# saver = tf.train.Saver()    # 声明tf.train.Saver类用于保存模型
# sess = tf.InteractiveSession()   #创建一个会话
# model(X_train , Y_train, X_test, Y_test)
# saver.save(sess, "modle/model.ckpt")  # 将模型保存到save/model文件





