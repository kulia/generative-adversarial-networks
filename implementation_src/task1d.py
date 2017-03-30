import regression.logistic_regression as lr
import numpy as np
import latex_helpers

var_latex_path = '../report_src/variables/'

def task1d():
    W_1 = np.array([[-0.1, 0.2, 0.1], [0.1, 0.4, 0.1], [0.0, -0.7, 0.2], [0.6, 0.3, -0.4]])
    W_2 = np.array([[0.3, -0.8, 0.1, 0.0], [0.0, 0.1, 0.2, 0.8], [-0.2, 0.7, 0.4, 0.1]])
    W_3 = np.array([[0.2], [0.1], [0.5], [0.4]])
    W = np.array([W_1, W_2, W_3])
    x_0 = np.array([1, -4, 0, 7])

    x_i = x_0
    iterator = 0
    for W_i in W:
        x_i = lr.compute_layer(x_i, W_i, lr.sigmoid)

        iterator += 1
        print(np.round_(x_i, decimals=2))
        latex_helpers.write_variable_to_latex(np.round_(x_i.T, decimals=2), 'x_{}'.format(iterator), var_latex_path)