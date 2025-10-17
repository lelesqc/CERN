import os
import sys
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import params_fcc
import functions as fn

par = params_fcc.Params()

def run_action_angle(mode):
    data = np.load(f"integrator/evolved_qp_{mode}.npz")

    q = data['q']
    p = data['p']


    if q.ndim == 1:
        n_steps = 1
        n_particles = q.shape[0]
        q = q.reshape((1, n_particles))
        p = p.reshape((1, n_particles))
    else:
        n_steps, n_particles = q.shape

    mask = np.empty(n_steps, dtype=object)
    avg_energies = np.zeros(n_steps)
    variances = np.zeros(n_steps)

    actions_list = np.zeros((n_steps, n_particles))

    x = np.zeros((n_steps, n_particles))
    y = np.zeros((n_steps, n_particles))
    
    for j in tqdm(range(n_particles)):
        for i in range(n_steps):
            h_0 = fn.H0_for_action_angle(q[i, j], p[i, j], par)
            kappa_squared = 0.5 * (1 + h_0 / (par.A**2))

            if 0 < kappa_squared < 1:
                Q = (q[i, j] + np.pi) / par.lambd
                P = par.lambd * p[i, j]

                action, theta = fn.compute_action_angle(kappa_squared, P)
                actions_list[i, j] = action 

                x[i, j] = np.sqrt(2 * action) * np.cos(theta)
                y[i, j] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(q[i, j]-np.pi)

    """    mask[i] = (x[i, :] + 0.2)**2 + y[i, :]**2 > 1.8**2
        q_masked = q[i, :][mask[i]]
        p_masked = p[i, :][mask[i]]
        energies = np.mean(fn.hamiltonian(q_masked, p_masked, par))
        avg_energies[i] = energies
        avg_energies[i] /= par.omega_rev

        variances[i] = np.var(p[i, :])

    times = np.linspace(0, par.dt * 18750, n_steps)
    plt.scatter(times, avg_energies, s=1)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$\mathcal{H} / \omega_\text{rev}$")
    #plt.savefig("../results/island/mean_energy_vs_time_fcc")
    plt.show()

    times = np.linspace(0, par.dt * 18750, n_steps-5)
    plt.scatter(times, variances[5:], s=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Variance")
    plt.yscale("log")
    plt.savefig("../results/island/variance_vs_time_fcc")
    plt.show()"""

    x = np.array(x)
    y = np.array(y)

    return x, y, actions_list

# --------------- Save results ----------------


if __name__ == "__main__":
    mode = sys.argv[1]
    x, y, actions_list = run_action_angle(mode)

    output_dir = "action_angle"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}_{par.sigma}.npz")
    np.savez(file_path, x=x, y=y, actions_list=actions_list)