import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt

import functions as fn
import params

par = params.Params()

def plot(mode):
    if mode == "evolution":
        data = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")
        integrator = np.load("integrator/evolved_qp_evolution.npz")

        x = data["x"]
        y = data["y"]
      
        """plt.hist(delta_E, bins=25, density=True, alpha=0.5, label="Simulazione")
        plt.plot(bin_centers, exp_func(bin_centers, T_fit, norm_fit), 'r-', lw=2, label=fr"Fit $T={T_fit:.2f}$")
        plt.title(f"Temperature: {T_fit:.1f}")
        plt.xlabel("Energies", fontsize=20)
        plt.ylabel("Probability Density", fontsize=20)
        #plt.yscale("log")
        #plt.plot(energies_neg, P_H, 'r-', lw=2, label="Teorica $e^{-(E-E_0)/T_{eff}}$")
        plt.show()
        """
        ps = np.load("./action_angle/phasespace_a0.050_nu0.80_extra.npz")
        x_ps = ps["x"]
        y_ps = ps["y"]

        plt.scatter(x_ps, y_ps, s=1, label="Phase space")
        plt.scatter(x, y, s=1, label="Distribution")
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("square")
        #plt.savefig("../results/phasespaces/ps_a0.050_nu0.80_with_damping_als.png")
        plt.show()

    if mode == "phasespace":
        action_angle = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")

        x = action_angle["x"]
        y = action_angle["y"]

        plt.scatter(x, y, s=1)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Phase space")
        #plt.savefig(f"../results/phasespaces/ps_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}_als.png", dpi=200)
        plt.show()


# ----------------------------------


if __name__ == "__main__":
    mode = sys.argv[1]

    plot(mode)
