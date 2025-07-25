import sys
import numpy as np
import matplotlib.pyplot as plt

import params as par

def plot(poincare_mode, n_particles, n_to_plot):
    a_start = par.a_lambda(par.T_percent)
    omega_start = par.omega_lambda(par.T_percent)
    a_end = par.a_lambda(par.T_tot)
    omega_end = par.omega_lambda(par.T_tot)

    a_start_str = f"{a_start:.3f}"
    omega_start_str = f"{omega_start:.2f}"
    a_end_str = f"{a_end:.3f}"
    omega_end_str = f"{omega_end:.2f}"

    str_title = f"a{a_start_str}-{a_end_str}_nu{float(omega_start_str)/par.omega_s:.2f}-{float(omega_end_str)/par.omega_s:.2f}_{n_particles}"

    data = np.load(f"action_angle/{poincare_mode}_{str_title}.npz")

    x = data['x']
    y = data['y']

    if poincare_mode in ["first", "last"]:
        plt.figure(figsize=(7,7))
        plt.scatter(x, y, s=3, label=r"Final distribution", alpha=1.0)
        plt.xlabel("X", fontsize=20)
        plt.ylabel("Y", fontsize=16)
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.legend(fontsize=20)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.show()

    elif poincare_mode == "all":
        n_sections = len(x) // n_particles
        idx_to_plot = np.linspace(0, n_sections-1, n_to_plot, dtype=int)

        _, axes = plt.subplots(1, n_to_plot, figsize=(4*n_to_plot, 4), sharex=True, sharey=True)
        if n_to_plot == 1:
            axes = [axes]
        for ax, i in zip(axes, idx_to_plot):
            ax.scatter(x[i*n_particles:(i+1)*n_particles], y[i*n_particles:(i+1)*n_particles], s=2)
            ax.set_xlabel("X")
            ax.set_ylabel("Y") 

        plt.suptitle(f"Poincaré sections for {poincare_mode} mode", fontsize=20)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust top to make room for the suptitle
        plt.show()

    elif poincare_mode == "none":
        plt.figure(figsize=(7,7))
        plt.scatter(x, y, s=3, label=r"Final distribution", alpha=1.0)
        print(x.shape)
        plt.xlabel("X", fontsize=20)
        plt.ylabel("Y", fontsize=16)
        plt.xlim(-15, 15)
        plt.ylim(-15, 15)
        plt.legend(fontsize=20)
        plt.tick_params(labelsize=18)
        plt.tight_layout()
        plt.show()

# -----------------------------------------


if __name__ == "__main__":
    poincare_mode = str(sys.argv[1])
    n_particles = int(sys.argv[2])
    n_to_plot = int(sys.argv[3])
    
    plot(poincare_mode, n_particles, n_to_plot)
