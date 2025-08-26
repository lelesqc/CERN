#%%

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import delayed

import params as par
import functions as fn
import action_angle as aa

base_dir = os.environ.get("BASE_DIR", "/mnt/c/Users/emanu/OneDrive - Alma Mater Studiorum Università di Bologna/CERN_data/code")
starting_data = np.load(base_dir + "/init_conditions/init_distribution.npz")
q_init = starting_data['q']
p_init = starting_data['p']

q = np.copy(q_init)
p = np.copy(p_init)

psi = 0
par.t = 0

n_particles = len(q_init)

q_ext = np.zeros((par.n_steps, n_particles))
p_ext = np.zeros((par.n_steps, n_particles))
sec_count = 0

times = []
psi_vals = []

while par.t < par.T_tot: 
    q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

    if np.cos(psi) > 1.0 - 1e-3:              
        q_ext[sec_count] = q
        p_ext[sec_count] = p
        times.append(par.t)
        psi_vals.append(psi)
        sec_count += 1

    psi += par.omega_lambda(par.t) * par.dt
    par.t += par.dt

q = q_ext[:sec_count]    # (poincare_found, n_particles)
p = p_ext[:sec_count]

np.savez(base_dir + "/actions_stuff/particle_data_phasespace.npz", q=q, p=p, times=times, psi_vals=psi_vals) 

#%% 
import numpy as np
import params as par
import functions as fn
from tqdm import tqdm

data = np.load(base_dir + "/actions_stuff/particle_data_phasespace.npz")
q = data['q'][:, ::100]
p = data['p'][:, ::100]
times = data['times']
psi_vals = data['psi_vals']

poincare_points = 250

inner_traj_q = np.zeros((int(n_particles / 100), q.shape[0], poincare_points), dtype=np.float16)    # i-th particle, j-th ext step, k-th int step
inner_traj_p = np.zeros((int(n_particles / 100), p.shape[0], poincare_points), dtype=np.float16)

psi_tracker = psi_vals

int_sec_count = 0

print(q.shape, inner_traj_p.shape)    # q: (885, 100)

for j in tqdm(range(q.shape[0])):
    q_temp = q[j, :]
    p_temp = p[j, :]
    psi_temp = psi_tracker[j]

    while int_sec_count < inner_traj_q.shape[2]:
        q_temp, p_temp = fn.integrator_step_fixed(q_temp, p_temp, psi_temp, par.a_lambda(times[j]), par.omega_lambda(times[j]), par.dt, fn.Delta_q_fixed, fn.dV_dq)
        
        if np.cos(psi_temp) > 1.0 - 1e-3:
            inner_traj_q[:, j, int_sec_count] = q_temp
            inner_traj_p[:, j, int_sec_count] = p_temp

            int_sec_count += 1

        psi_temp += par.omega_lambda(times[j]) * par.dt


#%%

# inner_traj_q: (n_particles, passi ext, passi int)

x = np.zeros((int(n_particles / 100), q.shape[0], poincare_points), dtype=np.float16)
y = np.zeros((int(n_particles / 100), q.shape[0], poincare_points), dtype=np.float16)

for i in tqdm(range(x.shape[0])):   # ciclo sulle particelle
    for j in range(x.shape[1]):     # ciclo sui punti esterni
        for k in range(x.shape[2]):    # ciclo sui punti interni
            h_0 = fn.H0_for_action_angle(inner_traj_q[i, j, k], inner_traj_p[i, j, k])
            kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
            if 0 < kappa_squared < 1:
                Q = (inner_traj_q[i, j, k] + np.pi) / par.lambd
                P = par.lambd * inner_traj_p[i, j, k]
                action, theta = fn.compute_action_angle(kappa_squared, P)
                x[i, j, k] = np.sqrt(2 * action) * np.cos(theta)
                y[i, j, k] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(inner_traj_q[i, j, k] - np.pi)
        
# Salvataggio dati
#np.savez("./actions_stuff/actions_first_part.npz", init_actions=init_actions)

#%%

print(x)


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
