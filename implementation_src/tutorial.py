import numpy as np
import tensorflow as tf

def tutorial():
    data_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    data_y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    X = tf.placeholder(tf.float32, shape=(None, 2))
    y = tf.placeholder(tf.float32, shape=(None, 1))

    weights =   { 'w1': tf.Variable(tf.random_uniform([2, 2], -1, 1))
                , 'b1': tf.Variable(tf.zeros([2]))
                , 'w2': tf.Variable(tf.random_uniform([2, 1], -1, 1))
                , 'b2': tf.Variable(tf.zeros([1]))
                }

    z1 = tf.add(tf.matmul(X, weights['w1']), weights['b1'])
    h1 = tf.sigmoid(z1)
    z2 = tf.add(tf.matmul(h1, weights['w2']), weights['b2'])
    y_hat = tf.sigmoid(z2)

    error = -tf.reduce_mean(tf.multiply(y, tf.log(y_hat)) + tf.multiply(1 - y, tf.log(1 - y_hat)))

    lr = 10.0

    optimiser = tf.train.GradientDescentOptimizer(lr).minimize(error, var_list=weights.values())

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        nb_epochs = 1000
        for epoch in range(nb_epochs):
            sess.run(optimiser, feed_dict={X: data_X, y: data_y})
            print(sess.run(error, feed_dict={X: data_X, y: data_y}))

