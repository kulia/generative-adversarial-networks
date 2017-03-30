import regression.logistic_regression as lr
import numpy as np

def task1d():
    W_1 = np.array([[-0.1, 0.2, 0.1], [0.1, 0.4, 0.1], [0.0, -0.7, 0.2], [0.6, 0.3, -0.4]])
    W_2 = np.array([[0.3, -0.8, 0.1, 0.0], [0.0, 0.1, 0.2, 0.8], [-0.2, 0.7, 0.4, 0.1]])
    W_3 = np.array([[0.2], [0.1], [0.5], [0.4]])
    W = np.array([W_1, W_2, W_3])
