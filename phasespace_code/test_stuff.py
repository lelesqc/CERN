import numpy as np
import matplotlib.pyplot as plt
import functions as fn
import os
import alphashape
from shapely.geometry import Polygon, MultiPolygon 
from scipy.special import ellipk

import params as par
from tqdm import tqdm

tunes = np.load("tune_analysis/tunes_results.npz")['tunes_list']
xy_data = np.load("../code/action_angle/last_a0.025-0.050_nu0.90-0.80_10000.npz")
integrator_data = np.load("../code/integrator/evolved_qp_last_10000.npz")

psi = integrator_data['psi']

x = xy_data['x']
y = xy_data['y']

mask_tunes_island = (tunes < 0.85) & (x**2 + y**2 > 2)
mask_tunes_center = ~mask_tunes_island 
tunes_center = tunes[mask_tunes_center]
tunes_island = tunes[mask_tunes_island]

idx_island = np.where(mask_tunes_island)[0]
idx_center = np.where(mask_tunes_center)[0]

is_island = np.zeros(x.shape, dtype=bool)
is_island[idx_island] = True

x_island = x[mask_tunes_island]
y_island = y[mask_tunes_island]
x_center = x[mask_tunes_center]
y_center = y[mask_tunes_center]

center_isl = (9.93, -0.27)
center_cen = (-0.95, 0.0)

positive_x_island_mask = x_island != 0
positive_x_center_mask = x_center != 0

x_island_positive = x_island[positive_x_island_mask]
y_island_positive = y_island[positive_x_island_mask]
x_center_positive = x_center[positive_x_center_mask]
y_center_positive = y_center[positive_x_center_mask]

#print(x_island_positive.shape, x_center_positive.shape)

xy = np.column_stack((x, y))
#xy = np.column_stack((x_island_positive, y_island_positive))

starting_points = xy

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

extra_steps = 5000
a = 0.05
omega_m = 0.8 * par.omega_s
t = 0.0

q_traj = np.empty((extra_steps, len(q_init)))
p_traj = np.empty((extra_steps, len(p_init)))
step_count = 0

while step_count < extra_steps:
    q += fn.Delta_q_fixed(p, psi, a, omega_m, par.dt/2)
    q = np.mod(q, 2 * np.pi)        
    t_mid = t + par.dt/2
    p += par.dt * fn.dV_dq(q)
    q += fn.Delta_q_fixed(p, psi, a, omega_m, par.dt/2)
    q = np.mod(q, 2 * np.pi)

    if np.cos(psi) > 1.0 - 1e-4:
        q_traj[step_count] = q
        p_traj[step_count] = p                    
        step_count += 1

    psi += omega_m * par.dt
    par.t += par.dt

q = q_traj[:step_count]
p = p_traj[:step_count]
n_particles = len(q_init)


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

#XY = np.zeros((x.shape[0], x.shape[1], 2), dtype=np.float16)

actions = []

for i in tqdm(range(x.shape[1])):  
    x_traj = x[:, i]
    y_traj = y[:, i]
    points = np.column_stack((x_traj, y_traj))

    points = np.unique(points, axis=0)

    if len(points) < 4:
        actions.append(np.nan)
        continue

    alpha = 0.3 if is_island[i] else 0.1    
    polygon = alphashape.alphashape(points, alpha)
    
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    area = polygon.area
    action = area / (2 * np.pi)
    actions.append(action)

    """if i % 2 == 0:
        if isinstance(polygon, Polygon):
            plt.plot(*polygon.exterior.xy, color='red', label='Concave Hull')
            plt.fill(*polygon.exterior.xy, alpha=0.3, color='green')
        elif isinstance(polygon, MultiPolygon):
            for poly in polygon.geoms:
                plt.plot(*poly.exterior.xy, color='red')
                plt.fill(*poly.exterior.xy, alpha=0.3, color='green')

        plt.scatter(points[:, 0], points[:, 1], s=1, color='blue')
        plt.title(f"Area: {area:.2f}")
        plt.show()"""


actions = np.array(actions)

np.savez("actions_analysis/final_actions_10000_test.npz", final_actions=actions)
