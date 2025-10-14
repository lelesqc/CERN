import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import functions as fn
import params_fcc

par = params_fcc.Params()

def plot(mode):
    if mode == "evolution":
        data = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")
        integrator = np.load("integrator/evolved_qp_evolution.npz")

        x = data["x"]
        y = data["y"]

        q = integrator['q']
        p = integrator['p']

        energies = np.array(fn.hamiltonian(q, p, par))
        energies_center = energies[energies < 0]
        energies_island = energies[energies > 0]

        q_center = q[energies < 0]
        p_center = p[energies < 0]

        q0 = np.mean(q_center)
        p0 = np.mean(p_center)

        h0_center = fn.hamiltonian(q0, p0, par)

        delta_E = energies_center - h0_center

        temp = np.mean(energies)
      
        """plt.hist(delta_E, bins=25, density=True, alpha=0.5, label="Simulazione")
        plt.plot(bin_centers, exp_func(bin_centers, T_fit, norm_fit), 'r-', lw=2, label=fr"Fit $T={T_fit:.2f}$")
        plt.title(f"Temperature: {T_fit:.1f}")
        plt.xlabel("Energies", fontsize=20)
        plt.ylabel("Probability Density", fontsize=20)
        #plt.yscale("log")
        #plt.plot(energies_neg, P_H, 'r-', lw=2, label="Teorica $e^{-(E-E_0)/T_{eff}}$")
        plt.show()
        """
        ps = np.load("./action_angle/phasespace_a0.050_nu0.80.npz")
        x_ps = ps["x"]
        y_ps = ps["y"]

        plt.scatter(x_ps, y_ps, s=1)
        plt.scatter(x, y, s=1)
        plt.show()

    if mode == "phasespace":
        action_angle = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")

        x = action_angle["x"]
        y = action_angle["y"]

        n_steps, n_particles = x.shape

        plt.scatter(x, y, s=10)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Phase space")
        #plt.savefig(f"./phasespaces/ps_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.png", dpi=200)
        plt.show()


# ----------------------------------


if __name__ == "__main__":
    mode = sys.argv[1]

    plot(mode)
