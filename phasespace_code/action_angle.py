import os
import sys
import numpy as np
from tqdm.auto import tqdm

import params as par
import functions as fn


def run_action_angle(mode):
    data = np.load(f"integrator/evolved_qp_{mode}.npz")

    q = data['q']
    p = data['p']

    n_steps, n_particles = q.shape

    actions_list = np.zeros((n_steps, n_particles))

    x = np.zeros((n_steps, n_particles))
    y = np.zeros((n_steps, n_particles))
    
    for j in tqdm(range(n_particles)):
        for i in range(n_steps):
            h_0 = fn.H0_for_action_angle(q[i, j], p[i, j])
            kappa_squared = 0.5 * (1 + h_0 / (par.A**2))

            if 0 < kappa_squared < 1:
                Q = (q[i, j] + np.pi) / par.lambd
                P = par.lambd * p[i, j]

                action, theta = fn.compute_action_angle(kappa_squared, P)
                actions_list[i, j] = action 

                x[i, j] = np.sqrt(2 * action) * np.cos(theta)
                y[i, j] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(q[i, j]-np.pi)

    x = np.array(x)
    y = np.array(y)

    actions_list = np.array(actions_list)

    return x, y, actions_list

# --------------- Save results ----------------


if __name__ == "__main__":
    mode = sys.argv[1]
    x, y, actions_list = run_action_angle(mode)

    output_dir = "action_angle"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{mode}_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")
    np.savez(file_path, x=x, y=y, actions_list=actions_list)