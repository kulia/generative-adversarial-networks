import numpy as np

def logistic_function(x_i, w):
    z = w.T @ x_i
    return np.divide(1.0, 1.0+np.exp(-z))


def derivative_cross_entropy_error(X, y, w):
    diff_error = np.zeros(3).reshape((3, 1))
    for x_i, y_i in zip(X, y):
        x_i = x_i.T
        sigma = logistic_function(x_i, w)
        diff_error += float(sigma-y_i)*x_i
    return diff_error


def gradient_descent_optimization(X, y, w, eta):
    return w-eta*derivative_cross_entropy_error(X, y, w)


def likelyhood_computation(X, y, w):
    l = 0
    for x_i, y_i in zip(X, y):
        x_i = x_i.T
        sigma = logistic_function(x_i, w)

        # To avoid log(0):
        if sigma == 0:
            sigma = 10**-10
        elif sigma == 1:
            sigma = 1-10**-10

        l+=y_i*np.log(sigma)+(1-y_i)*np.log(1-sigma)
    return l


def cross_entropy_error_function(X, y, w):
    n = X.shape[0]
    l = likelyhood_computation(X, y, w)
    return (1/n)*l
