#%%

import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import os
import params as par
from tqdm import tqdm

tunes = np.load("tune_analysis/tunes_results.npz")['tunes_list']
data = np.load("../code/action_angle/last_a0.025-0.050_nu0.90-0.80_10000.npz")
data_integrator = np.load("../code/integrator/evolved_qp_last_10000.npz")

q = data_integrator['q']
p = data_integrator['p']

x = data['x']
y = data['y']

mask_tunes_island = (tunes < 0.85) & (x**2 + y**2 > 2)
mask_tunes_center = ~mask_tunes_island 
tunes_center = tunes[mask_tunes_center]
tunes_island = tunes[mask_tunes_island]

x_island = x[mask_tunes_island]
y_island = y[mask_tunes_island]
x_center = x[mask_tunes_center]
y_center = y[mask_tunes_center]

x0 = -0.95
y0 = 0.0

psi = 0
q = q[mask_tunes_center]
p = p[mask_tunes_center]

n_points = 5000
n_particles = len(q)

plt.scatter(x_center, y_center, s=1, label='Center', color='blue')
plt.show()

#%%

#--------- integrator ------------

q_traj = np.zeros((n_points+1, n_particles), dtype=np.float16)
p_traj = np.zeros((n_points+1, n_particles), dtype=np.float16)

q_traj[0, :] = q
p_traj[0, :] = p

q = np.copy(q_traj[0, :])
p = np.copy(p_traj[0, :])

check_array = np.zeros(n_particles, dtype=bool)
angles = np.zeros((n_points+1, n_particles), dtype=np.float16)
angles[0, :] = np.arctan2(p, q - np.pi)
step = 0

while step < n_points:
    q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

    if np.cos(psi) > 1.0 - 1e-3:
        q_traj[step + 1, :] = q
        p_traj[step + 1, :] = p
        step += 1

    #angles[step + 1, :] = np.arctan2(p, q - np.pi)

    psi += par.omega_m * par.dt
    par.t += par.dt

#angles = angles[:step, :]
#angles = np.unwrap(angles, axis=0)

q_cut = q_traj
p_cut = p_traj

plt.scatter(q_cut[:, 10], p_cut[:, 10], s=1, label='Closed Trajectory', color='red')
plt.show()

#%%


#q_cut = []
#p_cut = []

"""for i in range(n_particles):
    mask = np.abs(angles[:, i] - angles[0, i]) >= 2 * np.pi
    if np.any(mask):
        idx = np.argmax(mask)
        q_cut.append(q_traj[:idx, i])
        p_cut.append(p_traj[:idx, i]) """


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

plt.scatter(x_list[:, 10], y_list[:, 10], s=1, label='Closed Trajectory')
plt.show()


# %%
