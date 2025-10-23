import os
import importlib
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

import functions as fn

params_module = os.environ.get("PARAMS_MODULE")
params = importlib.import_module(params_module)
par = params.Params()

machine = os.environ.get("MACHINE").lower()

def plot(mode, n_particles):
    if mode == "evolution":
        data = np.load(f"action_angle/evolution_{n_particles}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}_{machine}.npz")

        x = data["x"]
        y = data["y"]

        plt.scatter(x, y, s=1)
        plt.show()

    if mode == "phasespace":
        chi2_tot = []

        data = np.load(f"action_angle/phasespace_{n_particles}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}_{machine}.npz")
        data_qp = np.load(f"integrator/{mode}_qp_{n_particles}_{machine}.npz")

        x = data["x"]
        y = data["y"]

        q = data_qp["q"]
        p = data_qp["p"]

        avg_energies = np.zeros(q.shape[0])
        variances = np.zeros(q.shape[0])

        for i in range(q.shape[0]):
            #mask = x[i, :]**2 + y[i, :]**2 > 12
            mask = x[i, :] > 1.5
            q_mask = q[i, :][mask]
            p_mask = p[i, :][mask]

            energies = np.mean(fn.hamiltonian(q_mask, p_mask, par))
            print(energies)
            avg_energies[i] = energies
            variances[i] = np.var(p_mask)

        plt.scatter(x[-1, :], y[-1, :])
        plt.show()
        n_times = x.shape[0]

        energies = data["energies"]
        actions = data["actions"]

        bins = 100
        E0 = fn.hamiltonian(np.pi, 0, par)
        dof = bins - 1 - 1

        for i in range(energies.shape[0]):
            energies_i = energies[i]
            sorted_idx = np.argsort(actions[i, :])
            energies_i = energies_i[sorted_idx]
            actions_sorted_i = actions[i, :][sorted_idx]
            
            counts, bin_edges = np.histogram(actions[i, :], bins=bins, density=True)
            counts = gaussian_filter1d(counts, sigma=1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_width = bin_edges[1] - bin_edges[0]

            P_H = np.exp(- (np.interp(bin_centers, actions_sorted_i, energies_i) - E0) / (par.temperature))
            Z = trapezoid(P_H, bin_centers)           
            P_H /= Z 

            exp_counts = P_H * counts.sum() * bin_width

            epsilon = 1e-16

            chi2 = np.sum((counts - exp_counts)**2 / (exp_counts + epsilon)) / dof
            chi2_tot.append(chi2)

    plt.hist(actions[-1, :], bins=bins, density=True, alpha=0.5, label="Distr. of actions")
    plt.plot(bin_centers, P_H, label="Boltz. distribution")
    plt.title(rf"Reduced $\chi^2$: {chi2:.2f}")
    plt.legend()
    plt.xlabel("Actions")
    plt.ylabel("Frequency")
    plt.show()

    times = np.linspace(0, par.n_steps * par.dt, n_times)

    print(chi2_tot[-1])
    timez = np.linspace(0, times[-1], len(chi2_tot))
    plt.scatter(timez, chi2_tot, s=2)
    plt.xlabel("Time [s]")
    plt.yscale("log")
    plt.ylabel(r"Reduced $\chi^2$")
    plt.savefig("../results/resonance11/center/chi2_fcc")
    plt.show()

    """print(len(times), avg_energies.shape[0], q.shape[0])

    plt.scatter(times, avg_energies, s=1)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\langle \mathcal{H} \rangle$")
    plt.yscale("log")
    #plt.savefig("../results/resonance11/island/mean_energy_vs_time_fcc")
    plt.show()

    plt.scatter(times[2:], variances[2:], s=1)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\langle \delta^2 \rangle$")
    plt.yscale("log")
    #plt.savefig("../results/resonance11/island/variance_vs_time_fcc")
    plt.show()
    
    plt.scatter(x, y, s=1)
    plt.show()"""

# ----------------------------------


if __name__ == "__main__":
    mode = sys.argv[1]
    n_particles = int(sys.argv[2])

    plot(mode, n_particles)
