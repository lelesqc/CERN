import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import functions as fn
import params_fcc  as par

def plot(mode):
    if mode == "evolution":
        #data = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")
        integrator = np.load("integrator/evolved_qp_evolution.npz")

        #x = data["x"]
        #y = data["y"]

        q = integrator['q']
        p = integrator['p']

        energies = np.array(fn.hamiltonian(q, p))
        energies_center = energies[energies < 0]
        energies_island = energies[energies > 0]

        q_center = q[energies < 0]
        p_center = p[energies < 0]

        q0 = np.mean(q_center)
        p0 = np.mean(p_center)

        h0_center = fn.hamiltonian(q0, p0)

        delta_E = energies_center - h0_center

        temp = np.mean(energies)
        print(temp)
        #T_eff = np.mean(energies_neg)
        #P_H = np.exp(-(energies_neg - np.min(energies_neg)) / T_eff)

        #P_H /= scipy.integrate.trapezoid(P_H, energies_neg - np.mean(energies_neg))
    
        #counts, bin_edges = np.histogram(delta_E, bins=25, density=True)
        #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        #def exp_func(E, T, norm):
        #    return norm * np.exp(-E / T)
        
        #popt, pcov = curve_fit(exp_func, bin_centers, counts, p0=[np.std(delta_E), np.max(counts)])

        #T_fit, norm_fit = popt

        #print(T_fit)

        """plt.hist(delta_E, bins=25, density=True, alpha=0.5, label="Simulazione")
        plt.plot(bin_centers, exp_func(bin_centers, T_fit, norm_fit), 'r-', lw=2, label=fr"Fit $T={T_fit:.2f}$")
        plt.title(f"Temperature: {T_fit:.1f}")
        plt.xlabel("Energies", fontsize=20)
        plt.ylabel("Probability Density", fontsize=20)
        #plt.yscale("log")
        #plt.plot(energies_neg, P_H, 'r-', lw=2, label="Teorica $e^{-(E-E_0)/T_{eff}}$")
        plt.show()
        """

        #plt.scatter(x, y)
        plt.show()

    if mode == "phasespace":
        action_angle = np.load(f"action_angle/{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")

        x = action_angle["x"][::10]
        y = action_angle["y"][::10]

        print(x.shape)

        n_steps, n_particles = x.shape

        plt.scatter(x, y, s=10)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Phase space")
        plt.show()


# ----------------------------------


if __name__ == "__main__":
    mode = sys.argv[1]

    plot(mode)
