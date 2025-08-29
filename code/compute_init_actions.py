#%%

import os
import numpy as np
from tqdm import tqdm
#from tqdm_joblib import tqdm_joblib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import params as par
import functions as fn
import action_angle as aa


starting_data = np.load("init_conditions/init_distribution.npz")
q_init = starting_data['q']
p_init = starting_data['p']

idx = 42   #island
#idx = 59 #center

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

np.savez("actions_stuff/particle_data.npz", q=q, p=p, times=times, psi_vals=psi_vals)    

#%%

import numpy as np
import params as par
import functions as fn
import matplotlib.pyplot as plt
import alphashape
from shapely.geometry import Polygon
from tqdm import tqdm

data = np.load("actions_stuff/particle_data.npz")
q = data['q']
p = data['p']
times = data['times']
psi_vals = data['psi_vals']

inner_traj_q = np.zeros((par.n_steps, len(q)))
inner_traj_p = np.zeros((par.n_steps, len(p)))

psi_tracker = psi_vals

for i in tqdm(range(len(q))):
    q_temp = q[i]
    p_temp = p[i]
    psi_temp = psi_tracker[i]
    for j in range(par.n_steps):
        q_temp, p_temp = fn.integrator_step(q_temp, p_temp, psi_temp, times[i], par.dt, fn.Delta_q, fn.dV_dq)
        #if np.cos(psi_temp) > 1.0 - 1e-3:
        inner_traj_q[j, i] = q_temp
        inner_traj_p[j, i] = p_temp
        psi_temp += par.omega_lambda(times[i]) * par.dt

#%%

q_loops = [list(col[col != 0]) for col in inner_traj_q.T]
p_loops = [list(col[col != 0]) for col in inner_traj_p.T]

n_steps = [len(q_loop) for q_loop in q_loops]
actions = []
hulls = []

for j in tqdm(range(len(q_loops))):
    q_loop = q_loops[j]
    p_loop = p_loops[j]

    x = np.zeros(n_steps[j])
    y = np.zeros(n_steps[j])

    for i in range(n_steps[j]):
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

    alpha = 0.3
    hull = alphashape.alphashape(xy, alpha)
    
    if hull.geom_type == "MultiPolygon":
        alpha_low = 0.025  # Scegli tu il valore più basso
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

q_loops = [list(col[col != 0]) for col in inner_traj_q.T]
p_loops = [list(col[col != 0]) for col in inner_traj_p.T]

tunes_loops = []

for idx_loop in range(len(q_loops)):
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

print(tunes_loops)


# %%

data = np.load("actions_stuff/actions_particle_island.npz", allow_pickle=True)
actionz = data['actions']
hulls = data['hulls']
#tunes_loops = data['tunes']

np.savez(f"actions_stuff/actions_particle_island_{par.N_turn}.npz", actions=actionz, hulls=hulls, tunes=tunes_loops)

print(len(actionz), len(times), len(tunes_loops))

sc = plt.scatter(times, actionz, c=tunes_loops, s=1, cmap='viridis')
plt.scatter(times, actions, s=1)
plt.xlabel("Time (s)")
plt.ylabel("Action")
plt.title("Actions for Particle colored by Tune")
plt.colorbar(sc, label="Tune")
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
