from task1d import task1d
from tutorial import tutorial
from task2 import task2
from task3 import task3
import matplotlib.pyplot as plt
import matplotlib as mpl

var_latex_path = '../report_src/variables/'

if __name__ == '__main__':
    mpl.rc('font', family='Times New Roman')
    mpl.rcParams.update({'font.size': 22})
    mpl.rcParams.update({'text.usetex': True})

    # task1d()
    # tutorial()
    # task2()
    task3()

    plt.show()