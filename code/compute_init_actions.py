#%%
"""
# ------ Inizializzazione -------
starting_data = np.load("init_conditions/init_distribution.npz")
q_init = starting_data['q']
p_init = starting_data['p']

#idx = 42   #island
idx = 59 #center

q_init_particle = q_init[idx]
p_init_particle = p_init[idx]

q = np.copy(q_init_particle)
p = np.copy(p_init_particle)

psi = 0

# ------ Integratore -------
par.t = 0

q_loops = []
p_loops = []


while par.t < par.T_tot: 
    closed = False   

    q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

    q_loop = [q]
    p_loop = [p]

    steps = 0

    delta_angle = 0
    angle_0 = np.arctan2(p, q - np.pi)
    angle_prev = angle_0

    while not closed:
        q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)
        angle = np.arctan2(p, q - np.pi)

        dtheta = (angle - angle_prev + np.pi) % (2 * np.pi) - np.pi
        delta_angle += dtheta
        angle_prev = angle

        #if par.t > 0.007091:
        #    print(np.abs(delta_angle))

        if np.abs(delta_angle) >= 2 * np.pi:
            closed = True
            q_loops.append(np.array(q_loop))
            p_loops.append(np.array(p_loop))
            q = q_loop[0]
            p = p_loop[0]
            break

        q_loop.append(q.copy())
        p_loop.append(p.copy())

    #plt.scatter(q_loop, p_loop, s=1)
    #plt.show()
    

    if par.t > 0.00139:
        plt.scatter(q_loop, p_loop, s=1)
        plt.title(f"t = {par.t:.5f}", fontsize=12)
        plt.pause(0.01)


    psi += par.omega_lambda(par.t) * par.dt
    par.t += par.dt

    #print(par.t)


#plt.show()

print("finito")


#%%


n_steps = [len(q_loop) for q_loop in q_loops]
actions = []

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
    area = 0.5 * np.abs(np.dot(x_closed[:-1], y_closed[1:]) - np.dot(y_closed[:-1], x_closed[1:]))
    action_final = area / (2 * np.pi)
    actions.append(action_final)

print(actions)


# Salvataggio dati
np.savez("./actions_stuff/actions_first_part_test.npz", actions=actions)


#%%

times = np.linspace(0, par.T_tot, len(actions))
plt.scatter(times, actions, s=2)
plt.xlabel("Time (s)")
plt.ylabel("Action")
plt.title(f"Actions for Particle with Tune > 0.8")
plt.show()
"""
#%%
"""
import matplotlib.pyplot as plt
import numpy as np
import params as par

# ------ Ricostruzione risultati -------

tune_data = np.load("../phasespace_code/tune_analysis/tunes_results.npz")
data = np.load("./actions_stuff/actions_first_part.npz")
actions = data['actions']

times = np.linspace(0, par.T_tot, len(actions))

print(actions.shape)
plt.scatter(times, actions, s=2)
plt.xlabel("Time (s)")
plt.ylabel("Action")
plt.title(f"Actions for Particle with Tune > 0.8")
plt.show()

"""
#%%

import os
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import params as par
import functions as fn
import action_angle as aa


starting_data = np.load("init_conditions/init_distribution.npz")
q_init = starting_data['q']
p_init = starting_data['p']

#idx = 42   #island
idx = 59 #center

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
q = data['q'][::10]
p = data['p'][::10]
times = data['times'][::10]
psi_vals = data['psi_vals'][::10]

psi_tracker = 0

inner_traj_q = np.zeros((par.n_steps, len(q)))
inner_traj_p = np.zeros((par.n_steps, len(p)))

psi_tracker = psi_vals

for i in tqdm(range(len(q))):
    q_temp = q[i]
    p_temp = p[i]
    psi_temp = psi_tracker[i]
    for j in range(par.n_steps):
        q_temp, p_temp = fn.integrator_step(q_temp, p_temp, psi_temp, times[i], par.dt, fn.Delta_q, fn.dV_dq)
        if np.cos(psi_temp) > 1.0 - 1e-3:
            inner_traj_q[j, i] = q_temp
            inner_traj_p[j, i] = p_temp
        psi_temp += par.omega_lambda(times[i]) * par.dt

#%%

q_loops = [list(col[col != 0]) for col in inner_traj_q.T]
p_loops = [list(col[col != 0]) for col in inner_traj_p.T]

n_steps = [len(q_loop) for q_loop in q_loops]
actions = []

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
    area = hull.area
    action_final = area / (2 * np.pi)

    actions.append(action_final)

    plt.scatter(hull.exterior.xy[0], hull.exterior.xy[1], s=1, label=f"Loop {j+1}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Alpha Shape of Particle Trajectories")

    print(area)

plt.show()

#%%

# %%

plt.scatter(times, actions, s=2)
plt.xlabel("Time (s)")
plt.ylabel("Action")
plt.ylim(0, 10)
plt.title(f"Actions for Particle with Tune > 0.8")
plt.show()
# %%
