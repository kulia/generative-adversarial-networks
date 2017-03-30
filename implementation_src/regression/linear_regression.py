import numpy as np

def calculate_ordinary_least_squares(X, y):
    """
    Calculates ordinary least squares
    :param x: input vector x
    :param y: target vector y
    :return: single weight vector w
    """

    # Direct implementation of (9) from assignment text
    return (np.linalg.pinv(np.transpose(X)@X))@(np.transpose(X)@y)


def calculate_hyperplane_points(w, X):
    """
    Calculates a hyperplane based on weight vector w and input vector X. Capital X must not be confused with x
    :param w: weight vector
    :param X: input vector
    :return: hyperplane h(x)
    """

    x = np.transpose(X)
    return np.transpose(w)@x


def mean_squared_error(w, X, y):
    """
    Calculates the mean squar error
    :param w: weight vector
    :param X: Input set
    :param y: Output set
    :return: E_mse
    """
    # Direct implementation of (6) from assignment text
    # Each part represent one part of (6)

    N = X.shape[0]

    part_1 = (np.transpose(w)@np.transpose(X)) @ (X@w)
    part_2 = 2 * np.transpose(X@w)@y
    part_3 = np.transpose(y)@y

    return 1/N * (part_1-part_2+part_3)