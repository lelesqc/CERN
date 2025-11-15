#%%

import numpy as np
import matplotlib.pyplot as plt
import alphashape
import os
from tqdm.auto import tqdm

import params as par
import functions as fn

import warnings
warnings.filterwarnings("ignore")

base_dir="/mnt/c/Users/emanu/OneDrive - Alma Mater Studiorum Università di Bologna/CERN_data/code"

init_data = np.load(base_dir + "/init_conditions/init_distribution.npz")

q = init_data["q"][::100]
p = init_data["p"][::100]

psi = 0
par.t = 0

n_particles = int(q.shape[0])

q_sec = np.zeros((par.n_steps, n_particles))
p_sec = np.zeros((par.n_steps, n_particles))
sec_count = 0

times = []
psi_vals = []

while par.t < par.T_tot: 
    q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

    if np.cos(psi) > 1.0 - 1e-3:              
        q_sec[sec_count, :] = q
        p_sec[sec_count, :] = p
        times.append(par.t)
        psi_vals.append(psi)
        sec_count += 1

    psi += par.omega_lambda(par.t) * par.dt
    par.t += par.dt

q = q_sec[:sec_count, :]
p = p_sec[:sec_count, :]

#%%

plt.scatter(q, p)
plt.show()

#%%

poincare_points = 1000

inner_traj_q = np.zeros((n_particles, q.shape[0], poincare_points), dtype=np.float16)    # i-th particle, j-th ext step, k-th int step
inner_traj_p = np.zeros((n_particles, p.shape[0], poincare_points), dtype=np.float16)

psi_tracker = psi_vals

for j in tqdm(range(q.shape[0])):
    q_temp = q[j, :]
    p_temp = p[j, :]
    psi_temp = psi_tracker[j]
    t = times[j]

    int_sec_count = 0
    while int_sec_count < poincare_points:
        q_temp, p_temp = fn.integrator_step(q_temp, p_temp, psi_temp, t, par.dt, fn.Delta_q, fn.dV_dq)

        if np.cos(psi_temp) > 1.0 - 1e-3:
            inner_traj_q[:, j, int_sec_count] = q_temp
            inner_traj_p[:, j, int_sec_count] = p_temp

            int_sec_count += 1

        psi_temp += par.omega_lambda(times[j]) * par.dt


x = np.zeros((inner_traj_q.shape[0], inner_traj_q.shape[1], inner_traj_q.shape[2]), dtype=np.float16)
y = np.zeros((inner_traj_p.shape[0], inner_traj_p.shape[1], inner_traj_p.shape[2]), dtype=np.float16)

#%%

hulls = np.zeros((inner_traj_q.shape[0], inner_traj_q.shape[1]))
areas = np.zeros((inner_traj_q.shape[0], inner_traj_q.shape[1]))
actions = np.zeros((inner_traj_q.shape[0], inner_traj_q.shape[1]))

x_closed = np.zeros((x.shape[0], x.shape[1], x.shape[2] + 1), dtype=x.dtype)
y_closed = np.zeros((y.shape[0], y.shape[1], y.shape[2] + 1), dtype=y.dtype)

for i in tqdm(range(x.shape[0])):   # ciclo sulle particelle
    for j in range(x.shape[1]):     # ciclo sui punti esterni
        for k in range(x.shape[2]):    # ciclo sui punti interni
            h_0 = fn.H0_for_action_angle(inner_traj_q[i, j, k], inner_traj_p[i, j, k])
            kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
            if 0 < kappa_squared < 1:
                #Q = (inner_traj_q[i, j, k] + np.pi) / par.lambd
                P = par.lambd * inner_traj_p[i, j, k]
                action, theta = fn.compute_action_angle(kappa_squared, P)
                x[i, j, k] = np.sqrt(2 * action) * np.cos(theta)
                y[i, j, k] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(inner_traj_q[i, j, k] - np.pi)

        x_closed[i, j, :-1] = x[i, j, :]
        x_closed[i, j, -1] = x[i, j, 0]
        y_closed[i, j, :-1] = y[i, j, :]
        y_closed[i, j, -1] = y[i, j, 0]

        xy = np.vstack((x_closed[i, j, :], y_closed[i, j, :])).T

        x_traj = x[i, j, :]
        y_traj = y[i, j, :]
        
        # baricentro
        cx, cy = np.mean(x_traj), np.mean(y_traj)
        
        # ordino i punti per angolo polare rispetto al baricentro
        angoli = np.arctan2(y_traj - cy, x_traj - cx)
        idx = np.argsort(angoli)
        x_ord, y_ord = x_traj[idx], y_traj[idx]
        
        # formula shoelace
        areas[i, j] = 0.5 * np.abs(
            np.dot(x_ord, np.roll(y_ord, -1)) -
            np.dot(y_ord, np.roll(x_ord, -1))
        )

        actions[i, j] = areas[i, j] / (2 * np.pi) 

output_dir = "./stochastic_studies/actions"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
file_path = os.path.join(output_dir, "actions.npz")
np.savez(file_path, actions=actions)

# %%

data = np.load("./stochastic_studies/actions/actions.npz")
actions = data["actions"]
sorted_indices = np.argsort(actions[:, -1])[::-1]
sorted_actions = actions[:, -1][sorted_indices]
times = np.linspace(0, actions.shape[0], actions.shape[0])
plt.scatter(times, sorted_actions)
plt.show()

# %%
