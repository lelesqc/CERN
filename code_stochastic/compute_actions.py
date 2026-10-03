#%%

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import params_fcc as par
import functions as fn

import alphashape
import sys


starting_data = np.load("../phasespace_stochastic/integrator/evolution_qp_10000_fcc_use_for_relax.npz")
q_init = starting_data['q']
p_init = starting_data['p']


#idx = 2    # center
#idx = 10    # island

idx = 10

q_init_particle = q_init[::idx]
p_init_particle = p_init[::idx]

q = np.copy(q_init_particle)
p = np.copy(p_init_particle)

psi = 0
par.t = 0

q_sec = np.zeros((par.n_steps, q.shape[0]))
p_sec = np.zeros((par.n_steps, q.shape[0]))
sec_count = 0

times = []
psi_vals = []

print(par.T_tot)

while par.t < (par.T_tot / 5) : 
    q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

    if np.cos(psi) > 1.0 - 1e-3:              
        q_sec[sec_count, :] = np.copy(q)
        p_sec[sec_count, :] = np.copy(p)
        times.append(par.t)
        psi_vals.append(psi)
        sec_count += 1

    psi += par.omega_lambda(par.t) * par.dt
    par.t += par.dt

jump = 2
q_f = q_sec[:sec_count, :][::jump, :]
p_f = p_sec[:sec_count, :][::jump, :]

times = times[::jump]
psi_vals = psi_vals[::jump]

#np.savez(base_dir + "/actions_stuff/particle_data.npz", q=q, p=p, times=times, psi_vals=psi_vals)    

plt.scatter(q_f, p_f, s=1)
plt.show()

#%%

plt.scatter(q_f[-1, :], p_f[-1, :])
plt.show()

#%%

extra_steps = 200

par.D = 0
par.damp_rate = 0

print(par.D, par.damp_rate)

ext_steps = q_f.shape[0]
int_steps = extra_steps
n_particles = q_f.shape[1]

inner_traj_q = np.zeros((ext_steps, int_steps, q_f.shape[1]))
inner_traj_p = np.zeros((ext_steps, int_steps, q_f.shape[1]))

for i in tqdm(range(ext_steps)):
    q_temp = q_f[i, :]
    p_temp = p_f[i, :]
    psi_temp = psi_vals[i]
    a_fix = par.a_lambda(times[i])
    omega_m = par.omega_lambda(times[i])

    j = 0
    while j < int_steps:
        #q_temp, p_temp = fn.integrator_step_fixed(q_temp, p_temp, psi_temp, a_fix, omega_m, par.dt, fn.Delta_q_fixed, fn.dV_dq)
        q_temp, p_temp = fn.integrator_step(q_temp, p_temp, psi_temp, times[i], par.dt, fn.Delta_q, fn.dV_dq)

        if np.cos(psi_temp) > 1.0 - 1e-3:
            inner_traj_q[i, j, :] = np.copy(q_temp)
            inner_traj_p[i, j, :] = np.copy(p_temp)
            j += 1

        psi_temp += par.omega_lambda(times[i]) * par.dt


#%%

plt.scatter(inner_traj_q[10, :, 10], inner_traj_p[10, :, 10])
plt.show()

#%%

inner_traj_q_less = inner_traj_q[::10, :, :]
inner_traj_p_less = inner_traj_p[::10, :, :]

actions = []

for i in tqdm(range(inner_traj_p_less.shape[0])):
    for k in range(inner_traj_p_less.shape[2]):
        xy = np.vstack((inner_traj_q_less[i, :, k], inner_traj_p_less[i, :, k])).T

        points = xy

        alpha = 0.001
        hull = alphashape.alphashape(points, alpha)
        area = hull.area
        action_final = area / (2 * np.pi)
        actions.append(action_final)



#%%

for i in range(inner_traj_p_less.shape[0]):
    plt.hist(actions[(i*inner_traj_p_less.shape[0]): ((i+1)*(inner_traj_p_less.shape[0]))], bins=15)
    plt.show()

#%%

q_loops = [list(col) for col in inner_traj_q.T]
p_loops = [list(col) for col in inner_traj_p.T]

steps = [len(q_loop) for q_loop in q_loops]
actions = []
hulls = []
xy_list = []

x = np.zeros((inner_traj_q.shape[0], inner_traj_q.shape[1], inner_traj_q.shape[2]))
y = np.zeros((inner_traj_q.shape[0], inner_traj_q.shape[1], inner_traj_q.shape[2]))

for i in tqdm(range(x.shape[0])):
    for j in range(x.shape[1]):
        for k in range(x.shape[2]):
            h_0 = fn.H0_for_action_angle(inner_traj_q[i, j, k], inner_traj_p[i, j, k])
            h_0 += par.A**2
            kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
            if 0 < kappa_squared < 1:
                Q = (inner_traj_q[i, j, k] + np.pi) / par.lambd
                P = par.lambd * inner_traj_p[i, j, k]
                action, theta = fn.compute_action_angle(kappa_squared, P)
                x[i, j, k] = np.sqrt(2 * action) * np.cos(theta)
                y[i, j, k] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(inner_traj_q[i, j, k] - np.pi)

        """xy = np.vstack((x, y)).T

        points = xy_list
        alpha = 0.001
        hull = alphashape.alphashape(points, alpha)
        area = hull.area
        action_final = area / (2 * np.pi)
        actions.append(action_final)"""

    #xy_list.append(xy)


#%%

from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors

actions = []

j = 0
for j in range(len(q_loops)):
    #alpha = 0.04
    #hull = alphashape.alphashape(xy_list[j], alpha)

    points = xy_list[j]
    
    # Trova punti con pochi vicini (bordo)
    neighbors = NearestNeighbors(n_neighbors=10)
    neighbors.fit(points)
    distances, indices = neighbors.kneighbors(points)
    
    # I punti di bordo hanno distanze maggiori ai vicini
    mean_distances = np.mean(distances, axis=1)
    threshold = np.percentile(mean_distances, 80)  # top 20% più isolati
    boundary_mask = mean_distances > threshold
    boundary_points = points[boundary_mask]
    
    # Applica alpha-shape ai punti di bordo
    alpha = 0.001
    hull = alphashape.alphashape(boundary_points, alpha)


    if hull.geom_type == "MultiPolygon":
        print("entro")
        alpha_low = 0.001
        hull = alphashape.alphashape(xy_list[j], alpha_low)
    

    #np.savez(base_dir + f"/actions_stuff/particle_data_xy_{j}.npz",)
    
    #alpha_opt = alphashape.optimizealpha(xy)
    #hull = alphashape.alphashape(xy, alpha_opt)   

    hulls.append(hull)
    area = hull.area
    action_final = area / (2 * np.pi)
    actions.append(action_final)

    #plt.scatter(xy_list[j][:, 0], xy_list[j][:, 1], s=2, label="Traiettoria")
    #plt.plot(hull.exterior.xy[0], hull.exterior.xy[1], color='red', label="Alpha shape")
    #plt.xlabel("X")
    #plt.ylabel("Y")
    #plt.title("Alpha Shape of Particle Trajectories")
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

actions = np.array(actions)
times = np.array(times)

print(times[-1])

# Maschere
mask_times = times > 3.2
mask_actions = (actions > 1.55) & (actions < 1.75)

# Costruisci la maschera finale:
final_mask = (~mask_times) | (mask_times & mask_actions)

plt.plot(times , actions)
plt.xlabel("Time [s]")
plt.ylabel("J")
#plt.ylim(-0.1, 2.0)
plt.show()


#%%

import matplotlib as mpl

# Verifica tutte le impostazioni font
print("Font family:", mpl.rcParams['font.family'])
print("Font size:", mpl.rcParams['font.size'])
print("Font specifico:", mpl.rcParams['font.sans-serif'])
print("Mathtext font:", mpl.rcParams['mathtext.fontset'])

#%%


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
