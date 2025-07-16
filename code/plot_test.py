import numpy as np
import matplotlib.pyplot as plt

def plot_test():
    data = np.load("integrator/evolved_qp_last.npz")

    q = data['q']
    p = data['p']

    plt.scatter(q, p)
    plt.show()

if __name__ == "__main__":
    plot_test()

