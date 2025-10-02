#%%

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
from scipy.integrate import trapezoid

import params_fcc as par
import functions as fn

import warnings
warnings.filterwarnings("ignore")

base_dir="/mnt/c/Users/emanu/OneDrive - Alma Mater Studiorum Università di Bologna/CERN_data/code"

init_data = np.load("./init_conditions/qp_10000.npz")

q = init_data["q"]
p = init_data["p"]

psi = 0
par.t = 0

n_particles = int(q.shape[0])

q_sec = np.zeros((par.n_steps // 50, n_particles), dtype=np.float32)
p_sec = np.zeros((par.n_steps // 50, n_particles), dtype=np.float32)
sec_count = 0

times = []
step = 0
for step in tqdm(range(par.n_steps)): 
    q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

    if np.cos(psi) > 1.0 - 1e-3:              
        q_sec[sec_count, :] = q
        p_sec[sec_count, :] = p
        times.append(par.t)

        sec_count += 1

    psi += par.omega_m * par.dt
    par.t += par.dt

q = q_sec[:sec_count, :]
p = p_sec[:sec_count, :]

#%%
# shape di q e p: (n_punti, n_particelle)

n_times = q.shape[0]
#n_times = 10
step = int(len(times)/n_times)

actions = np.empty((n_times, n_particles))
energies = []
kappas = []

for i, idx in enumerate(range(0, len(times), step)):
    h_0 = fn.H0_for_action_angle(q[idx, :], p[idx, :])
    kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
    kappas.append(kappa_squared)

    actions[i, :], _ = fn.compute_action_angle(kappa_squared, 1)
    
    h0_of_I = 2 * par.A**2 * (kappa_squared - 1/2)
    energies.append(h0_of_I)

# energies è una lista di n_times elementi, ogni elemento è una lista
# contenente l'energia delle 10'000 particelle per quell'istante di tempo

E0 = fn.hamiltonian(np.pi, 0)

noise_D = (par.gamma / par.beta**2 * np.sqrt(2 * par.damp_rate * par.h * par.eta * par.Cq / par.radius))**2
damping_factor = 2 * par.damp_rate / par.beta**2
temperature = noise_D / (2 * damping_factor)


#%%

from scipy.special import rel_entr
from scipy.stats import chisquare

kl_div_tot = []
chi2_tot = []

for i in range(n_times):
    nan_indices = np.where(np.isnan(actions[i, :]))[0]
    if nan_indices.size > 0:
        print(f"NaN in actions[{i},:] agli indici: {nan_indices}")
        print("Valori corrispondenti di q:", q[i, :][nan_indices])
        print("Valori corrispondenti di p:", p[i, :][nan_indices])
        print(kappas[i][nan_indices])
    energies_i = energies[i]
    sorted_idx = np.argsort(actions[i, :])
    energies_i = energies_i[sorted_idx]
    actions_sorted_i = actions[i, :][sorted_idx]

    # istogramma
    hist, bin_edges = np.histogram(actions[i, :], bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # curva teorica
    P_H = np.exp(- (np.interp(bin_centers, actions_sorted_i, energies_i) - E0) / (par.k_lele_fcc * temperature))
    Z = trapezoid(P_H, bin_centers)           
    P_H /= Z 


    epsilon = 1e-12
    kl_div = np.sum(rel_entr(hist + epsilon, P_H + epsilon))
    kl_div_tot.append(kl_div)

    chi2 = np.sum((hist - P_H)**2 / (P_H + epsilon))
    chi2_tot.append(chi2)

    """plt.hist(actions[i, :], bins=100, density=True, alpha=0.5, label="Empirico")
    plt.plot(bin_centers, P_H, label="Teorico")
    plt.title(f"KL Divergence: {kl_div:.2e}")
    plt.legend()
    plt.show()
    print(f"KL Divergence: {kl_div:.2e}")"""

plt.hist(actions[-1, :], bins=100, density=True, alpha=0.5, label="Distr. of actions")
plt.plot(bin_centers, P_H, label="Boltz. distribution")
plt.title(f"KL Divergence: {chi2:.2f}")
plt.legend()
plt.xlabel("Actions")
plt.ylabel("Frequency")
plt.show()

times = np.linspace(0, par.dt * par.n_steps, n_times)

#kl_div_tot = np.array(kl_div_tot)

plt.scatter(times[:n_times//8], chi2_tot[:n_times//8], s=2)
#plt.title(r"Kullback-Leibler divergence $\rho(I)$ - $\rho_{Boltz}(H(I))$ for FCC-ee")
#plt.ylabel("KL divergence")
#plt.xlabel("Time [s]")
plt.show()

#%%


"""
    for idx in range(len(energies)):
        q_mean = np.mean(q[idx, :])
        p_mean = np.mean(p[idx, :])
        E0 = fn.hamiltonian(q_mean, p_mean)
        E_max = np.max(energies[idx])
        T_th = temperature
        T_emp = np.mean(energies[idx] - E0)

        energies_pts = np.linspace(E0 - E0, E_max - E0, 500)
        P_H = np.exp(-energies_pts / T_emp)  
        Z = trapezoid(P_H, energies_pts)           
        P_H /= Z       
        
        plt.hist(energies[idx] - E0, bins=70, density=True)
        plt.title(f"Plot n. {idx}")
        #plt.plot(energies_pts, P_H)
        #plt.yscale("log")
        #plt.scatter(q[idx, :], p[idx, :])
        #plt.scatter(q_mean, p_mean, s = 30)
        plt.show()"""

"""
output_dir = "./stochastic_studies/actions"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
file_path = os.path.join(output_dir, "actions.npz")
np.savez(file_path, actions=actions) """

# %%

"""
data = np.load("./stochastic_studies/actions/actions.npz")
actions = data["actions"]
sorted_indices = np.argsort(actions[:, -1])[::-1]
sorted_actions = actions[:, -1][sorted_indices]
times = np.linspace(0, actions.shape[0], actions.shape[0])
plt.scatter(times, sorted_actions)
plt.show()"""