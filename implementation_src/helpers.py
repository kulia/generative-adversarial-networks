# -*- coding: utf-8 -*-
#
# helpers.py: Includes a set of rudimentary helper functions for the
# fourth assignment in TDT4173.
#
import os

import numpy as np

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
import scipy.ndimage

import matplotlib


def load_data(train_path, test_path):
    """Load and return the classification dataset from assignment 4.

    Can work for other CSV files, but assumes that there is only one
    target value.

    Parameters
    ----------
    train_path : str
    test_path : str
    """
    # Load training data
    train = np.genfromtxt('./{}'.format(train_path), delimiter=',')
    X_train = train[:, :train.shape[1]-1]
    y_train = train[:, train.shape[1]-1:]

    # Load test data
    test = np.genfromtxt('./{}'.format(test_path), delimiter=',')
    X_test = test[:, :test.shape[1]-1]
    y_test = test[:, test.shape[1]-1:]

    return X_train, y_train, X_test, y_test

def plot_data(fig, ax, x, y, y_hat=None):
    x1 = np.array(x[:, 0])
    x2 = np.array(x[:, 1])
    y = np.array(y)

    # x1_i = np.linspace(0, 1, 0.01)
    # x2_i = np.linspace(0, 1, 0.01)
    # # yi = griddata((x1, x2), y, (x1_i, x2_i))

    # X1, X2 = np.meshgrid(x1, x2)
    # yi = plt.mlab.bivariate_normal(x1, x2, y)

    # print('Test: ', len(x1_i), len(x2_i), len(yi))

    # meshgrid = scipy.ndimage.zoom((x1, x2, y), 3)
    # surf = ax.contour(meshgrid, 100, cmap=cm.jet)

    ax.scatter(x1, x2, c=y, cmap=cm.jet, alpha=1)

    if not y_hat is None:
        ax.scatter(x1, x2, c=y_hat, cmap=cm.jet, alpha=0.5)

def load_mnist_tf(path='./mnist'):
    """Download and return the MNIST dataset using TensorFlow.

    Parameter
    ---------
    path : str
        The location of where you would like the download MNIST.
    """
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets(path, one_hot=True)


def create_dir(path):
    """Ensure that the input path points to an existing directory.

    Parameter
    ---------
    path : str
    """
    if not os.path.exists(path):
        os.makedirs(path)


def plot_samples(samples):
    """Return a plot of the input samples in a grid.

    The number of samples and sample size must be divisible by 2.

    Parameter
    ---------
    samples : numpy.ndarray
        Must have exactly two dimensions. The first is the sample
        number, while the second is the sample itself (a vector of
        numbers).
    """
    assert samples.shape[0] % 2 == 0,\
        ('Number of samples is not divisible by 2.')
    assert len(samples[0]) % 2 == 0,\
        ('Sample size is not divisible by 2.')

    grid_size = int(np.sqrt(samples.shape[0]))
    img_size = int(np.sqrt(len(samples[0])))

    figure = plt.figure(figsize=(grid_size, grid_size))
    grid = matplotlib.gridspec.GridSpec(grid_size, grid_size)
    grid.update(hspace=0.1, wspace=0.1)

    for idx, sample in enumerate(samples):
        ax = plt.subplot(grid[idx])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(sample.reshape(img_size, img_size), cmap=plt.cm.gray)

    return figure
