import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import params as par
import functions as fn
import action_angle as aa

starting_data = np.load("../code/init_conditions/init_distribution.npz")
q_init = starting_data['q']
p_init = starting_data['p']

print(q_init.shape, p_init.shape)

n_particles = len(q_init)

q_traj = np.zeros((par.n_steps, n_particles), dtype=np.float16)
p_traj = np.zeros((par.n_steps, n_particles), dtype=np.float16)

q_traj[0, :] = q_init
p_traj[0, :] = p_init

q = np.copy(q_traj[0, :])
p = np.copy(p_traj[0, :])

psi = 0

check_array = np.zeros(n_particles, dtype=bool)
angles = np.zeros((par.n_steps, n_particles), dtype=np.float16)
angles[0, :] = np.arctan2(p, q - np.pi)

# ------ integrator -------

for step in tqdm(range(par.n_steps // 10)):
    q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

    angles[step + 1, :] = np.arctan2(p, q - np.pi)

    q_traj[step + 1, :] = q
    p_traj[step + 1, :] = p

    psi += par.omega_m * par.dt
    par.t += par.dt

angles = angles[:step, :]
angles = np.unwrap(angles, axis=0)

q_cut = []
p_cut = []

for i in range(n_particles):
    mask = np.abs(angles[:, i] - angles[0, i]) >= 2 * np.pi
    if np.any(mask):
        idx = np.argmax(mask)
        q_cut.append(q_traj[:idx, i])
        p_cut.append(p_traj[:idx, i])   

# ------ cartesian -------

x_list = []
y_list = []

for q, p in tqdm(zip(q_cut, p_cut), total=len(q_cut)):
        n_steps = len(q)
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        actions = np.zeros(n_steps)
        for i in range(n_steps):
            h_0 = fn.H0_for_action_angle(q[i], p[i])
            kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
            if 0 < kappa_squared < 1:
                Q = (q[i] + np.pi) / par.lambd
                P = par.lambd * p[i]
                action, theta = fn.compute_action_angle(kappa_squared, P)
                actions[i] = action
                x[i] = np.sqrt(2 * action) * np.cos(theta)
                y[i] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(q[i] - np.pi)
        x_list.append(x)
        y_list.append(y)

init_actions = []

for x, y in zip(x_list, y_list):
    # Chiudi il poligono aggiungendo il primo punto in fondo
    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])
    # Shoelace formula
    area = 0.5 * np.abs(np.dot(x_closed[:-1], y_closed[1:]) - np.dot(y_closed[:-1], x_closed[1:]))
    action = area / (2 * np.pi)
    init_actions.append(action)

init_actions = np.array(init_actions)
print(f"Initial actions:", np.round(init_actions, 1))
#os.makedirs("actions_analysis", exist_ok=True)
#np.savez("actions_analysis/init_actions_10000.npz", init_actions=init_actions)
