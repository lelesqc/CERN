import os
import importlib
import sys
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import functions as fn

params_module = os.environ.get("PARAMS_MODULE")
params = importlib.import_module(params_module)
par = params.Params()

machine = os.environ.get("MACHINE").lower()

def run_action_angle(mode, n_particles):
    data = np.load(f"integrator/{mode}_qp_{n_particles}_{machine}.npz")

    q = data['q']
    p = data['p']

    if q.ndim == 1:
        n_steps = 1
        q = q.reshape((1, n_particles))
        p = p.reshape((1, n_particles))
    else:
        n_steps = q.shape[0]

    actions_list = np.zeros((n_steps, n_particles))
    energies = np.zeros((n_steps, n_particles))

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
                energies[i, j] = h_0

                x[i, j] = np.sqrt(2 * action) * np.cos(theta)
                y[i, j] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(q[i, j]-np.pi)

    x = np.array(x)
    y = np.array(y)

    return x, y, actions_list, energies

# --------------- Save results ----------------


if __name__ == "__main__":
    mode = sys.argv[1]
    n_particles = int(sys.argv[2])
    x, y, actions_list, energies = run_action_angle(mode, n_particles)

    output_dir = "action_angle"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{mode}_{n_particles}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}_{machine}.npz")
    np.savez(file_path, x=x, y=y, actions=actions_list, energies=energies)