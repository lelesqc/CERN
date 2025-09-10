import os
import sys
import numpy as np
from tqdm.auto import tqdm

import params as par
import functions as fn

base_dir = os.environ["BASE_DIR"]

def run_action_angle(poincare_mode, n_particles):
    base_dir = os.environ["BASE_DIR"]

    data = np.load(base_dir + f"/integrator/evolved_qp_{poincare_mode}_{n_particles}.npz")

    q = data['q']
    p = data['p']

    if poincare_mode in ["all", "none"]:
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

    elif poincare_mode in ["first", "last"]:
        actions_list = []
        theta_list = []
        sign_list = []

        for i in tqdm(range(len(q))):
            h_0 = fn.H0_for_action_angle(q[i], p[i])
            kappa_squared = 0.5 * (1 + h_0 / (par.A**2))

            if 0 < kappa_squared < 1:
                #Q = (q[i] + np.pi) / par.lambd
                P = par.lambd * p[i]

                action, theta = fn.compute_action_angle(kappa_squared, P)
                actions_list.append(action)
                theta_list.append(theta)
                sign_list.append(np.sign(q[i]-np.pi))
            elif kappa_squared > 1:
                print(kappa_squared)

        actions_list = np.array(actions_list)
        theta_list = np.array(theta_list)

        x = np.sqrt(2 * np.array(actions_list)) * np.cos(theta_list)
        y = - np.sqrt(2 * np.array(actions_list)) * np.sin(theta_list) * np.array(sign_list)

    return x, y, actions_list


# --------------- Save results ----------------


if __name__ == "__main__":
    poincare_mode = sys.argv[1]
    n_particles = int(sys.argv[2])

    x, y, actions_list = run_action_angle(poincare_mode, n_particles)

    a_start = par.a_lambda(par.T_percent)
    omega_start = par.omega_lambda(par.T_percent)
    a_end = par.a_lambda(par.T_tot)
    omega_end = par.omega_lambda(par.T_tot)
    
    a_start_str = f"{a_start:.3f}"
    omega_start_str = f"{omega_start:.2f}"
    a_end_str = f"{a_end:.3f}"
    omega_end_str = f"{omega_end:.2f}"

    str_title = f"a{a_start_str}-{a_end_str}_nu{float(omega_start_str)/par.omega_s:.2f}-{float(omega_end_str)/par.omega_s:.2f}"

    output_dir = base_dir + "/action_angle"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{poincare_mode}_{str_title}_{n_particles}.npz")
    np.savez(file_path, x=x, y=y, actions=actions_list)

    print(f"Data saved in {file_path}")