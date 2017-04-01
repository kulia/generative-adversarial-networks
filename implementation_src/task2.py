import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import helpers

default_path = '../data/{}.csv'
train_path = default_path.format('cl-test')
test_path = default_path.format('cl-test')

def task2():
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)

    # Define model entry-points
    X = tf.placeholder(tf.float32, shape=(None, 2))
    y = tf.placeholder(tf.float32, shape=(None, 1))

    # Create weights and gather them in a dictionary
    weights = {'w1': tf.Variable(tf.random_uniform([2, 2], -1, 1)),
               'b1': tf.Variable(tf.zeros([2])),
               'w2': tf.Variable(tf.random_uniform([2, 1], -1, 1)),
               'b2': tf.Variable(tf.zeros([1]))}

    # Define model as a computational graph
    z1 = tf.add(tf.matmul(X, weights['w1']), weights['b1'])
    h1 = tf.sigmoid(z1)
    z2 = tf.add(tf.matmul(h1, weights['w2']), weights['b2'])
    y_hat = tf.sigmoid(z2)

    # Define error functions
    error = - tf.reduce_mean(tf.multiply(y, tf.log(y_hat)) +
                             tf.multiply(1 - y, tf.log(1 - y_hat)))

    # Specify which optimiser to use (`lr` is the learning rate)
    lr = 10.0
    optimiser = tf.train.GradientDescentOptimizer(lr).minimize(
        error, var_list=weights.values())

    # Generate Op that initialises global variables in the graph
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Initialise variables and start the session
        sess.run(init)

        # Run a set number of epochs
        nb_epochs = 1000
        for epoch in range(nb_epochs):
            sess.run(optimiser, feed_dict={X: X_train, y: y_train})

            # Print out some information every nth iteration
            if epoch % 100 == 0:
                err = sess.run(error, feed_dict={X: X_train, y: y_train})
                print('Epoch: ', epoch, '\t Error: ', err)

        # Print out the final predictions
        predictions = sess.run(y_hat, feed_dict={X: X_train, y: y_train})
        print('\nFinal XOR function predictions:')
        for idx in range(len(predictions)):
            print('{} ~> {:.2f}'.format(X_train[idx], predictions[idx][0]))

        # helpers.plot_curves('label1', X_train, 'label2', y_train)
        helpers.plot_samples(X_train)
        plt.show()

        del sess
        sys.exit(0)