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

q = np.copy(q_init)
p = np.copy(p_init)

psi = 0
par.t = 0

n_particles = len(q_init)

q_sec = np.zeros((par.n_steps, n_particles))
p_sec = np.zeros((par.n_steps, n_particles))
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

np.savez("actions_stuff/particle_data_phasespace.npz", q=q, p=p, times=times, psi_vals=psi_vals) 


#%%

import numpy as np
import params as par
import functions as fn
from tqdm import tqdm

data = np.load("actions_stuff/particle_data.npz")
q = data['q']
p = data['p']
times = data['times']
psi_vals = data['psi_vals']

poincare_points = 250

inner_traj_q = np.zeros((n_particles, len(q), poincare_points))    # i-th particle, j-th ext step, k-th int step
inner_traj_p = np.zeros((n_particles, len(p), poincare_points))

psi_tracker = psi_vals

for i in tqdm(range(n_particles)):
    for j in tqdm(range(len(q))):
        q_temp = q[j]
        p_temp = p[j]
        psi_temp = psi_tracker[j]
        for k in range(par.n_steps):
            q_temp, p_temp = fn.integrator_step(q_temp, p_temp, psi_temp, times[j], par.dt, fn.Delta_q, fn.dV_dq)
            if np.cos(psi_temp) > 1.0 - 1e-3:
                inner_traj_q[i, j, k] = q_temp
                inner_traj_p[i, j, k] = p_temp
            psi_temp += par.omega_lambda(times[j]) * par.dt



    
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
#np.savez("./actions_stuff/actions_first_part.npz", init_actions=init_actions)


#%%

times = np.linspace(0, par.T_percent, len(actions))
plt.scatter(times, actions, s=2)
plt.xlabel("Time (s)")
plt.ylabel("Action")
plt.title(f"Actions for Particle with Tune > 0.8")
plt.show()



#%%

import matplotlib.pyplot as plt
import numpy as np
import params as par

# ------ Ricostruzione risultati -------

tune_data = np.load("../phasespace_code/tune_analysis/tunes_results.npz")
data = np.load("./actions_stuff/actions_first_part.npz")
init_actions = data['actions']

tunes = tune_data['tunes_list']
mask = tunes < 0.82
indici_particelle = np.where(mask)[0]
print(indici_particelle[:10])


idx_particle = 999

#plt.xlabel("Time (s)")
#plt.ylabel("Action")
#plt.title(f"Actions for Particle with Tune > 0.8")
#plt.show()

# %%
