import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import helpers
from latex_helpers import write_variable_to_latex


default_path = '../data/{}.csv'
train_path = default_path.format('cl-test')
test_path = default_path.format('cl-test')
batch_size_test = 1

path_to_figure = '../report_src/figures/mlp/'
var_latex_path = '../report_src/variables/'

def task2():

    model = Model(train_path, test_path)

    learning_rate = 2.0
    write_variable_to_latex(learning_rate, 'lr', var_latex_path)

    optimiser, learning_curve, testing_curve = train_and_test_model(model, lr=learning_rate)

    visualize_results(learning_curve, testing_curve)

    print(model.y_hat)

def visualize_results(learning_curve, testing_curve):
    learning_curve = 100 * learning_curve
    testing_curve = 1-testing_curve
    testing_curve = 100 * testing_curve

    fig = plt.figure()
    ax1 = plt.subplot2grid((1, 1), (0, 0))

    epocs = np.arange(len(learning_curve))
    ax1.plot(epocs, learning_curve, c='k', label='Train')
    ax1.plot(batch_size_test*np.arange(len(testing_curve)), testing_curve, c='r', label='Test')
    ax1.legend(prop={'family': 'Times'}, loc='lower left')

    ax1.set_xlim([0, len(learning_curve)])

    # plt.title('Learning curve')
    plt.ylabel('Cross-entropy error [\%]')
    plt.xlabel('Number of epochs')

    plt.subplots_adjust(left=0.13, right=0.95, top=0.92, bottom=0.16)

    plt.savefig(path_to_figure + 'cross_entropy_error.pdf', format='pdf', dpi=1000)


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

def train_and_test_model(model, lr=10, nb_epochs=1000):
    # Specify which optimiser to use (`lr` is the learning rate)
    optimiser = tf.train.GradientDescentOptimizer(lr).minimize(
        model.error, var_list=model.weights.values())

    # Generate Op that initialises global variables in the graph
    init = tf.global_variables_initializer()

    learning_curve = np.array([])
    testing_curve = np.array([])

    with tf.Session() as sess:
        # Initialise variables and start the session
        sess.run(init)

        # Run a set number of epochs
        for epoch in range(nb_epochs):
            sess.run(optimiser, feed_dict={model.X: model.X_train, model.y: model.y_train})

            error_temp_train = sess.run(model.error, feed_dict={model.X: model.X_train, model.y: model.y_train})
            learning_curve = np.append(learning_curve, error_temp_train)

            if not epoch % batch_size_test:
                correct_prediction = tf.equal(tf.round(model.y_hat), model.y_test)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                error_temp_test = accuracy.eval({model.X: model.X_test, model.y: model.y_test})
                testing_curve = np.append(testing_curve, error_temp_test)

    return optimiser, learning_curve, testing_curve