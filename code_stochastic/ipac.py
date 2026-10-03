#%%

import numpy as np
import matplotlib.pyplot as plt
import params_fcc as par
import functions as fn
import alphashape
from tqdm.auto import tqdm

a = par.a_lambda(par.T_tot)
omega_m = par.omega_lambda(par.T_tot)
steps = 250
n_particles = 10000

q_traj = np.zeros((steps, n_particles))
p_traj = np.zeros((steps, n_particles))

init_data = np.load("../phasespace_stochastic/integrator/evolution_qp_10000_fcc_relaxed_1.00.npz")
q = init_data["q"]
p = init_data["p"]

step = 1 
while step < steps:
    q, p = fn.integrator_step_fixed(q, p, psi, a, omega_m, par.dt, fn.Delta_q_fixed, fn.dV_dq)

    if np.cos(psi) > (1.0 - 1e-4):
        q_traj[step, :] = np.copy(q)
        p_traj[step, :] = np.copy(p)

        step += 1

    psi += omega_m * par.dt


x = np.zeros((steps, q_traj.shape[1]))
y = np.zeros((steps, q_traj.shape[1]))

for j in tqdm(range(n_particles)):
    for i in range(x.shape[1]):
        h_0 = fn.H0_for_action_angle(q[i, j], p[i, j], par)
        kappa_squared = 0.5 * (1 + h_0 / (par.A**2))

        if 0 < kappa_squared < 1:
            Q = (q[i, j] + np.pi) / par.lambd
            P = par.lambd * p[i, j]

            action, theta = fn.compute_action_angle(kappa_squared, P)
            
            x[i, j] = np.sqrt(2 * action) * np.cos(theta)
            y[i, j] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(q[i, j]-np.pi)
    
x = np.array(x)
y = np.array(y)

x0 = np.mean(x_fin)
y0 = np.mean(y_fin)

X = np.vstack([x_fin - x0, y_fin - y0])  # shape (2, N)
Sigma = np.cov(X)                       # (2, 2)

X_points = X.T                          # (N, 2)
det_Sigma = np.linalg.det(Sigma)

Sigma_inv = np.linalg.inv(Sigma)

emittance = np.sqrt(det_Sigma)

print(emittance)

linear_J_cen = 0.5 * np.sqrt(det_Sigma) * np.einsum('ni,ij,nj->n', X_points, Sigma_inv, X_points)

print(linear_J_cen)

# %%
