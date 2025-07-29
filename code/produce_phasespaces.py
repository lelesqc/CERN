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

# ------ Inizializzazione -------
starting_data = np.load("init_conditions/init_distribution.npz")
q_init = starting_data['q']
p_init = starting_data['p']

q_init_particle = q_init
p_init_particle = p_init

q = np.copy(q_init_particle)
p = np.copy(p_init_particle)

psi = 0

# ------ Integratore -------
par.t = 0

q_loops_all = []
p_loops_all = []

n_sections = 10

steps_section = 500

for idx in tqdm(range(len(q_init))):
    q = np.copy(q_init[idx])
    p = np.copy(p_init[idx])
    psi = 0
    par.t = 0

    q_loops = []
    p_loops = []

    section = 0

    while par.t < par.T_percent: 
        closed = False   
        q_loop = [q]
        p_loop = [p]
        steps = 0
        delta_angle = 0
        angle_0 = np.arctan2(p, q - np.pi)
        angle_prev = angle_0

        for i in range(steps_section):
            q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)
            
            if np.cos(psi) > 1.0 - 1e-3:              
                q_sec[sec_count] = q
                p_sec[sec_count] = p
                sec_count += 1

        psi += par.omega_lambda(par.t) * par.dt
        par.t += par.dt

    q_loops_all.append(q_loops)
    p_loops_all.append(p_loops)

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
