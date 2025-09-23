#%%

import numpy as np
import scipy
import matplotlib.pyplot as plt
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

q_sec = np.zeros((par.n_steps // 10, n_particles), dtype=np.float32)
p_sec = np.zeros((par.n_steps // 10, n_particles), dtype=np.float32)
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
# shape di q e p: (n_punti, n_particelle)

n_times = 10
step = int(len(times)/n_times)

actions = np.zeros((n_times, n_particles))
energies = []
kappa_squared_list = []

for i, idx in enumerate(range(0, len(times), step)):
    t = times[idx]
    h_0 = fn.H0_for_action_angle(q[idx, :], p[idx, :])
    energies.append(h_0)
    kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
    actions[i, :], _ = fn.compute_action_angle(kappa_squared, 1)
    h0_of_I = 2 * par.A**2 * (kappa_squared - 1/2)
    #energies.append(h0_of_I)

actions = np.array(actions)

#%%
from scipy.integrate import trapezoid

noise_D = (par.gamma / par.beta**2 * np.sqrt(2 * par.damp_rate * par.h * par.eta * par.Cq / par.radius))**2
damping_factor = 2 * par.damp_rate / par.beta**2
temperature = noise_D / (2 * damping_factor)

for idx in range(len(energies)):
    E_min = np.min(energies[idx])
    E_max = np.max(energies[idx])
    T = temperature
    #T = np.mean(energies[idx] - E_min)

    energies_pts = np.linspace(E_min - E_min, E_max - E_min, 500)
    P_H = np.exp(-energies_pts / T)  
    Z = trapezoid(P_H, energies_pts)           
    P_H /= Z       

    plt.hist(energies[idx] - np.min(energies[idx]), bins=100, density=True)
    plt.title(f"Plot n. {idx}")
    plt.plot(energies_pts, P_H)
    #plt.yscale("log")
    plt.show()


#%%

plt.scatter(q, p)
plt.show()
print(T)


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