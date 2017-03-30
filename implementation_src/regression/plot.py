import numpy as np
from matplotlib import cm

def plot_training_set_2D(training_set,fig, ax, z_offset=0):
    x1 = np.array(training_set[:, 0])
    x2 = np.array(training_set[:, 1])
    y = np.array(training_set[:, 2])
    scatt = ax.scatter(x1, x2, c=y, cmap=cm.jet, vmin=0, vmax=1)
    return fig, ax, scatt


def plot_training_set_3D(training_set,fig, ax, z_offset=0):
    x1 = np.array(training_set[:, 0])
    x2 = np.array(training_set[:, 1])
    y = np.array(training_set[:, 2])
    scatt = ax.scatter(x1, x2, y+z_offset, c=y, cmap=cm.jet, linewidth=0)
    return fig, ax, scatt


def plot_hyperplane_2D(X, hyperplane, fig, ax):
    X_cord, Y_cord, Z_cord = calculate_hyperplane_meshgrid(X, hyperplane)
    surf = ax.pcolormesh(X_cord, Y_cord, Z_cord, cmap=cm.jet, vmin=0, vmax=1)
    return fig, ax


def plot_hyperplane_3D(X, hyperplane, fig, ax):
    X_cord, Y_cord, Z_cord = calculate_hyperplane_meshgrid(X, hyperplane)
    ax.plot_surface(X_cord, Y_cord, Z_cord, cmap=cm.jet, linewidth=0, antialiased=False)
    return fig, ax


def set_xy_direction(ax):
    ax.view_init(elev=90, azim=270)
    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])



def format_hyplerplane_plot(ax):
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(False)


def calculate_hyperplane_meshgrid(X, hyperplane):
    p0 = np.array([np.array(np.array(X)[0][1]), np.array(np.array(X)[0][2]), np.array(np.array(hyperplane)[0][0])])
    p1 = np.array([np.array(np.array(X)[1][1]), np.array(np.array(X)[1][2]), np.array(np.array(hyperplane)[0][1])])
    p2 = np.array([np.array(np.array(X)[2][1]), np.array(np.array(X)[2][2]), np.array(np.array(hyperplane)[0][2])])

    # These two vectors are in the plane
    v1 = p2 - p0
    v2 = p1 - p0

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p2)

    x = np.linspace(0, 1)
    y = np.linspace(0, 1)
    X_cord, Y_cord = np.meshgrid(x, y)

    Z_cord = (d - a * X_cord - b * Y_cord) / c

    return X_cord, Y_cord, Z_cord