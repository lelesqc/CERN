#%%

import numpy as np
import matplotlib.pyplot as plt
import alphashape
import os
from tqdm.auto import tqdm

import params as par
import functions as fn

import warnings
warnings.filterwarnings("ignore")

base_dir="/mnt/c/Users/emanu/OneDrive - Alma Mater Studiorum Università di Bologna/CERN_data/code"

init_data = np.load("./init_conditions/qp.npz")

q = init_data["q"]
p = init_data["p"]

psi = 0
par.t = 0

n_particles = int(q.shape[0])

q_sec = np.zeros((par.n_steps, n_particles))
p_sec = np.zeros((par.n_steps, n_particles))
sec_count = 0

times = []
step = 0
while step < par.n_steps: 
    q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

    if np.cos(psi) > 1.0 - 1e-3:              
        q_sec[sec_count, :] = q
        p_sec[sec_count, :] = p
        times.append(par.t)
        sec_count += 1

    psi += par.omega_m * par.dt
    par.t += par.dt

    step += 1

q = q_sec[:sec_count, :]
p = p_sec[:sec_count, :]

#%%

# shape di q e p: (1'000, 10'000) = (n_punti, n_particelle)

n_times = 10
actions = np.zeros((n_times, n_particles))

for i, idx in enumerate(range(0, len(times), 100)):
    t = times[idx]
    h_0 = fn.H0_for_action_angle(q[idx, :], p[idx, :])
    kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
    actions[i, :], _ = fn.compute_action_angle(kappa_squared, 1)

actions = np.array(actions)

#%%

noise_D = (par.gamma / par.beta**2 * np.sqrt(par.damp_rate * par.h * par.eta * par.Cq / par.radius))**2
temperature = par.beta**2 * noise_D / (2 * par.damp_rate)
print(temperature)

temperature = np.mean(h_0)
print(temperature)
th_curve = np.exp(- h_0 / temperature)

#print(- h_0 / temperature)

plt.hist(actions[0, :], bins=100)
plt.yscale("log")
#plt.plot(actions[9, :], th_curve)
plt.show()

#plt.hist(actions[9, :], bins=100)
#plt.show()

#%%

output_dir = "./stochastic_studies/actions"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
file_path = os.path.join(output_dir, "actions.npz")
np.savez(file_path, actions=actions)

# %%

data = np.load("./stochastic_studies/actions/actions.npz")
actions = data["actions"]
sorted_indices = np.argsort(actions[:, -1])[::-1]
sorted_actions = actions[:, -1][sorted_indices]
times = np.linspace(0, actions.shape[0], actions.shape[0])
plt.scatter(times, sorted_actions)
plt.show()