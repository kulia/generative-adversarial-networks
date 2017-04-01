import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import helpers

default_path = '../data/{}.csv'
train_path = default_path.format('cl-test')
test_path = default_path.format('cl-test')

def task2():

    model = Model(train_path, test_path)

    learning_rate = 2.0
    learning_curve = train_model(model, lr=learning_rate)

    #Visualize
    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    helpers.plot_data(fig, ax1, model.X_train, model.y_train)

    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    plt.plot(np.arange(len(learning_curve)), learning_curve)

class Model:
    def __init__(self, train_path, test_path):
        self.X = tf.placeholder(tf.float32, shape=(None, 2))
        self.y = tf.placeholder(tf.float32, shape=(None, 1))
        self.X_train, self.y_train, self.X_test, self.y_test = helpers.load_data(train_path, test_path)
        self.weights = self.set_default_weghts()
        self.y_hat, self.error = self.build_model()

    def build_model(self):
        # Define model as a computational graph
        z1 = tf.add(tf.matmul(self.X, self.weights['w1']), self.weights['b1'])
        h1 = tf.sigmoid(z1)
        z2 = tf.add(tf.matmul(h1, self.weights['w2']), self.weights['b2'])
        y_hat = tf.sigmoid(z2)

        # Define error functions
        error = - tf.reduce_mean(tf.multiply(self.y, tf.log(y_hat)) +
                                 tf.multiply(1 - self.y, tf.log(1 - y_hat)))

        return y_hat, error

    def set_default_weghts(self):
        return {'w1': tf.Variable(tf.random_uniform([2, 3], -1, 1))
            , 'b1': tf.Variable(tf.zeros([3]))
            , 'w2': tf.Variable(tf.random_uniform([3, 1], -1, 1))
            , 'b2': tf.Variable(tf.zeros([1]))
                }

def train_model(model, lr=10):
    # Specify which optimiser to use (`lr` is the learning rate)
    optimiser = tf.train.GradientDescentOptimizer(lr).minimize(
        model.error, var_list=model.weights.values())

    # Generate Op that initialises global variables in the graph
    init = tf.global_variables_initializer()

    learning_curve = np.array([])

    with tf.Session() as sess:
        # Initialise variables and start the session
        sess.run(init)

        # Run a set number of epochs
        nb_epochs = 10000
        for epoch in range(nb_epochs):
            sess.run(optimiser, feed_dict={model.X: model.X_train, model.y: model.y_train})

            error_temp = sess.run(model.error, feed_dict={model.X: model.X_train, model.y: model.y_train})
            learning_curve = np.append(learning_curve, error_temp)

    return learning_curve


def visualize_results():
    pass