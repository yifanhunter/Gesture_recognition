# Author:yifan
import tensorflow as tf
import math
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow.contrib.slim.nets as nets
from TF_GDR02_Data import load_dataset
import matplotlib.image as mpimg # mpimg 用于读取图片


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset() #取数据
X_train = X_train_orig/255
X_test = X_test_orig/255

#定义计算算损失函数
def compute_cost(out, y):
    logits = tf.transpose(out) #转置
    labels = tf.transpose(y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost

#X(1080, 64, 64, 3)
#Y(1, 1080)
def random_mini_batches(X, Y, mini_batch_size = 4, seed = 0):
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
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    Indices = np.arange(m)
    np.random.shuffle(Indices)   #返回序号出错乱的数据0-1080
    # print(Indices)
    shuffled_X = X[Indices,:,:,:]
    Y = np.array(Y)
    shuffled_Y = Y[Indices]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[ k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[ num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches




#定义模型保存地址，batch_sizes设置的小一点训练效果更好，将当前目录下的tfrecord文件放入列表中:
# save_dir = r"./train_image_63.model"
# batch_size_ = 16
x = tf.placeholder(tf.float32, [None, 64, 64, 3])
y_ = tf.placeholder(tf.float32, [None])


# 将label值进行onehot编码
one_hot_labels = tf.one_hot(indices=tf.cast(y_, tf.int32), depth=6)
pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=6, is_training=True)
pred = tf.reshape(pred, shape=[-1, 6])


#参数
minibatch_size = 4
lr = tf.Variable(0.0001, dtype=tf.float32)
nums_epoch = 200

# 定义损失函数和优化器
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=one_hot_labels))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# 准确度
a = tf.argmax(pred, 1)
b = tf.argmax(one_hot_labels, 1)
correct_pred = tf.equal(a, b)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

seed = 0
# saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(nums_epoch):
        epoch_cost = 0
        m = X_train.shape[0]
        num_minibatches = int(m / minibatch_size)  # 1080/32  表示训练minibatches的数量
        seed = seed + 1  # 每次seed不一样，不同的sess号会不同
        minibatches = random_mini_batches(X_train, Y_train_orig, minibatch_size, seed)
        for minibatch in minibatches:
            (b_image, b_label ) = minibatch  # 切成小块的minibatch (32, 64, 64, 3)   (32,)
            _, loss_, y_t, y_p, a_, b_ = sess.run([optimizer, loss, one_hot_labels, pred, a, b], feed_dict={x: b_image, y_: b_label})

            # print(y_t)  #真是值得one-hot
            # print(y_p)  #推测值得one-hot
            # print(b_label)  #实际值
        print('第{}轮:, train_loss: {}'.format(seed, loss_))

        pathtest1 = 'data/test/0/18.jpg'
        pathtest1 = mpimg.imread(pathtest1) / 255
        test1_set_x = np.array([pathtest1])  # 数据类型为( 64, 64, 3)
        ytest_p1 = sess.run(a, feed_dict={x: test1_set_x})
        print(ytest_p1)

        pathtest2 = 'data/test/1/4.jpg'
        pathtest2 = mpimg.imread(pathtest2) / 255
        test2_set_x = np.array([pathtest2])  # 数据类型为( 64, 64, 3)
        ytest_p2 = sess.run(a, feed_dict={x: test2_set_x})
        print(ytest_p2)

    # 保存模型
    saver = tf.train.Saver()  # 声明tf.train.Saver类用于保存模型
    saver.save(sess, "GR_model03//cnn_model.ckpt")

    ytest_p = sess.run(a, feed_dict={x: X_test})
    ytrain_p = sess.run(a, feed_dict={x: X_train})
    Acctrain = np.mean(ytrain_p == Y_train_orig)
    print(Acctrain)

    Acctest = np.mean(ytest_p == Y_test_orig)
    print(Acctest)
    Conf_Mat = confusion_matrix(ytest_p,Y_test_orig)
    print(Conf_Mat)
    print(ytest_p)

    # 载入图片数据
    pathtest1 = 'data/test/0/18.jpg'
    pathtest1 = mpimg.imread(pathtest1) / 255
    test1_set_x = np.array([pathtest1])  # 数据类型为( 64, 64, 3)
    ytest_p1 = sess.run(a, feed_dict={x: test1_set_x})
    print(ytest_p1)

    pathtest2 = 'data/test/1/4.jpg'
    pathtest2 = mpimg.imread(pathtest2) / 255
    test2_set_x = np.array([pathtest2])  # 数据类型为( 64, 64, 3)
    ytest_p2 = sess.run(a, feed_dict={x: test2_set_x})
    print(ytest_p2)



