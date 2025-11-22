#%%

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from scipy.special import ellipk

import params_fcc as par
import functions as fn

base_dir = os.environ["BASE_DIR"]

xy_data = np.load(base_dir + f"/action_angle/last_a0.025-0.050_nu0.90-0.80_10000.npz")
tunes_data = np.load(base_dir + f"/../phasespace_code/tune_analysis/tunes_results.npz")
integrator_data = np.load(base_dir + "/integrator/evolved_qp_last_5000.npz")

psi = integrator_data['psi']

# ---------- initial conditions ----------

x = xy_data['x']
y = xy_data['y']
#tunes_list = tunes_data['tunes_list']

mask = (x**2 + y**2 > 2)
#tunes_list = tunes_list[mask]
x = x[mask]
y = y[mask]

center = (9.93, -0.27)

mask_x = (x >= center[0]) & (x <= np.max(x))

x = x[mask_x]
y = y[mask_x]

xy = np.column_stack((x, y))

starting_points = fn.pts_in_section(xy, center[0], center[1], np.max(x), 1.0)

kappa_squared_list = np.empty(starting_points.shape[0])
Omega_list = np.empty(starting_points.shape[0])
Q_list = np.empty(starting_points.shape[0])
P_list = np.empty(starting_points.shape[0])

action, theta = fn.compute_action_angle_inverse(starting_points[:, 0], starting_points[:, 1])
for i, act in enumerate(action):
        h_0 = fn.find_h0_numerical(act)
        kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
        kappa_squared_list[i] = kappa_squared
        Omega_list[i] = np.pi / 2 * (par.A / ellipk(kappa_squared))

for i, (angle, freq, k2) in enumerate(zip(theta, Omega_list, kappa_squared_list)):
    Q, P = fn.compute_Q_P(angle, freq, k2)
    Q_list[i] = Q
    P_list[i] = P

phi, delta = fn.compute_phi_delta(Q_list, P_list)
phi = np.mod(phi, 2 * np.pi) 
q_init = np.array(phi)
p_init = np.array(delta)
q = q_init.copy()
p = p_init.copy()

a = 0.05
omega_m = 0.8 * par.omega_s
extra_steps = 4096


# ---------- integrator -----------

#%%

t = par.t

q_traj = np.empty((extra_steps, len(q_init)))
p_traj = np.empty((extra_steps, len(p_init)))
step_count = 0
for _ in range(extra_steps):
    q += fn.Delta_q_fixed(p, psi, a, omega_m, par.dt/2)
    q = np.mod(q, 2 * np.pi)        
    t_mid = t + par.dt/2
    p += par.dt * fn.dV_dq(q)
    q += fn.Delta_q_fixed(p, psi, a, omega_m, par.dt/2)
    q = np.mod(q, 2 * np.pi)

    if np.cos(psi) > 1.0 - 1e-3:
        q_traj[step_count] = q
        p_traj[step_count] = p                    
        step_count += 1

    psi += omega_m * par.dt
    par.t += par.dt

q = q_traj[:step_count]
p = p_traj[:step_count]
n_particles = len(q_init)


# --------- action-angle -----------

#%%

x = np.zeros((len(q), n_particles))
y = np.zeros((len(q), n_particles))
for j in tqdm(range(n_particles)):
    for i in range(len(q)):
        h_0 = fn.H0_for_action_angle(q[i, j], p[i, j])
        kappa_squared = 0.5 * (1 + h_0 / (par.A**2))

        if 0 < kappa_squared < 1:
            Q = (q[i, j] + np.pi) / par.lambd
            P = par.lambd * p[i, j]

            action, theta = fn.compute_action_angle(kappa_squared, P)

            x[i, j] = np.sqrt(2 * action) * np.cos(theta)
            y[i, j] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(q[i, j]-np.pi)

x = np.array(x)
y = np.array(y)

#output_dir = "tune_stuff"
#if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#np.savez("./tune_stuff/island_particles.npz", x=x, y=y)


# ---------- save and plot -----------

#%%

plt.scatter(x, y, s=1, label="Phase Space for final distr.", alpha=1.0)
plt.show()

