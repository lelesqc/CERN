#%%

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import params as par
import functions as fn

import alphashape

import sys

os.environ["BASE_DIR"] = "/mnt/c/Users/emanu/OneDrive - Alma Mater Studiorum Università di Bologna/CERN_data/code"
base_dir = os.environ["BASE_DIR"]

starting_data = np.load(base_dir + "/init_conditions/init_distribution.npz")
q_init = starting_data['q']
p_init = starting_data['p']

idx = 42    #island
idx = 100    #center

q_init_particle = q_init[idx]
p_init_particle = p_init[idx]

q = np.copy(q_init_particle)
p = np.copy(p_init_particle)

psi = 0
par.t = 0

q_sec = np.zeros(par.n_steps)
p_sec = np.zeros(par.n_steps)
sec_count = 0

times = []
psi_vals = []

while par.t < par.T_tot: 
    q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

    if np.cos(psi) > 1.0 - 1e-3:              
        q_sec[sec_count] = q
        p_sec[sec_count] = p
        times.append(par.t)
        psi_vals.append(psi)
        sec_count += 1

    psi += par.omega_lambda(par.t) * par.dt
    par.t += par.dt

q = q_sec[:sec_count]
p = p_sec[:sec_count]

np.savez(base_dir + "/actions_stuff/particle_data.npz", q=q, p=p, times=times, psi_vals=psi_vals)    


#%%

inner_traj_q = np.zeros((par.n_steps, len(q)))
inner_traj_p = np.zeros((par.n_steps, len(p)))

psi_tracker = psi_vals

extra_steps = 1000

for i in tqdm(range(len(q))):
    q_temp = q[i]
    p_temp = p[i]
    psi_temp = psi_tracker[i]

    j = 0
    while j < extra_steps:
        q_temp, p_temp = fn.integrator_step(q_temp, p_temp, psi_temp, times[i], par.dt, fn.Delta_q, fn.dV_dq)

        if np.cos(psi_temp) > 1.0 - 1e-3:
            inner_traj_q[j, i] = q_temp
            inner_traj_p[j, i] = p_temp
            j += 1

        psi_temp += par.omega_lambda(times[i]) * par.dt

#%%

q_loops = [list(col[col != 0]) for col in inner_traj_q.T]
p_loops = [list(col[col != 0]) for col in inner_traj_p.T]

steps = [len(q_loop) for q_loop in q_loops]
actions = []
hulls = []

for j in tqdm(range(len(q_loops))):
    q_loop = q_loops[j]
    p_loop = p_loops[j]

    x = np.zeros(steps[j])
    y = np.zeros(steps[j])

    for i in range(steps[j]):
        h_0 = fn.H0_for_action_angle(q_loop[i], p_loop[i])
        kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
        if 0 < kappa_squared < 1:
            Q = (q_loop[i] + np.pi) / par.lambd
            P = par.lambd * p_loop[i]
            action, theta = fn.compute_action_angle(kappa_squared, P)
            x[i] = np.sqrt(2 * action) * np.cos(theta)
            y[i] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(q_loop[i] - np.pi)

    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])

    xy = np.vstack((x_closed, y_closed)).T

    alpha = 0.2
    hull = alphashape.alphashape(xy, alpha)
    
    if hull.geom_type == "MultiPolygon":
        alpha_low = 0.025
        hull = alphashape.alphashape(xy, alpha_low)

    hulls.append(hull)
    area = hull.area
    action_final = area / (2 * np.pi)
    actions.append(action_final)

    #plt.scatter(hull.exterior.xy[0], hull.exterior.xy[1], s=1, label=f"Loop {j+1}")
    #plt.xlabel("X")
    #plt.ylabel("Y")
    #plt.title("Alpha Shape of Particle Trajectories")

    #print(area)

#plt.show()

#%%

"""
tunes_loops = []

for idx_loop in tqdm(range(len(q_loops))):
    q_loop = np.array(q_loops[idx_loop])
    p_loop = np.array(p_loops[idx_loop])

    z = (q_loop - np.mean(q_loop)) - 1j * p_loop
    z_normalized = z / np.abs(z)
    angles = np.angle(z_normalized, deg=False)
    angles_unwrapped = np.unwrap(angles)
    delta_angles = np.diff(angles_unwrapped)
    delta_angles = np.abs(delta_angles) / (2 * np.pi) * par.N
    tune = fn.birkhoff_average(delta_angles)
    tunes_loops.append(tune)
"""
    
# %%

#sc = plt.scatter(times, actions, c=tunes_loops, s=1, cmap='viridis')
plt.scatter(times, actions, s=1)
plt.xlabel("Time (s)")
plt.ylabel("Action")
plt.title("Actions for Particle colored by Tune")
#plt.colorbar(sc, label="Tune")
plt.show()

"""
print(hull)
plt.xlabel("Time (s)")
plt.ylabel("Action")
#plt.ylim(0, 7)
plt.title(f"Actions for Particle with Tune > 0.8")
if hasattr(hull, "exterior"):
        from matplotlib.patches import Polygon as MplPolygon
        fig, ax = plt.subplots()
        patch = MplPolygon(np.array(hull.exterior.coords), closed=True, facecolor='lightblue', edgecolor='b', alpha=0.5)
        ax.add_patch(patch)
        ax.scatter(hull.exterior.xy[0], hull.exterior.xy[1], s=1)

        #ax.scatter(xy[:, 0], xy[:, 1], s=2, color='red')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Alpha Shape of Particle Trajectories")
        plt.show()
#plt.show() """

# %%
