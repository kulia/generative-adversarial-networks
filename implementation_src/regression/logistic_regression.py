import numpy as np

def compute_layer(x_i, W_i, activation_function):
    return activation_function(weight_multiplication(x_i, W_i))

def sigmoid(z):
    return np.divide(1.0, 1.0 + np.exp(-z))

def weight_multiplication(x_i, W_i):
    return W_i.T @ x_i