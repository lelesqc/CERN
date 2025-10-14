#%%

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
from scipy.integrate import trapezoid

import params_fcc
import functions as fn

import warnings
warnings.filterwarnings("ignore")

par = params_fcc.Params()

base_dir="/mnt/c/Users/emanu/OneDrive - Alma Mater Studiorum Università di Bologna/CERN_data/code"

init_data = np.load("./init_conditions/qp_10000.npz")

q = init_data["q"]
p = init_data["p"]

plt.scatter(q, p)
plt.show()

psi = 0
par.t = 0

n_particles = int(q.shape[0])

q_sec = np.zeros((par.n_steps // 50, n_particles), dtype=np.float32)
p_sec = np.zeros((par.n_steps // 50, n_particles), dtype=np.float32)
sec_count = 0

times = []
step = 0
for step in tqdm(range(par.n_steps)): 
    q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq, par)

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

# DA USARE PER TROVARE VARIANZA PARTICELLE DENTRO ISOLA

thr = 25
energies = np.empty(q.shape[0], dtype=object)
p_vars = np.zeros(q.shape[0]-thr)

q_mask = np.empty(q.shape[0]-thr, dtype=object)
p_mask = np.empty(p.shape[0]-thr, dtype=object)

for i in tqdm(range(energies.shape[0])):
    energies_i = fn.hamiltonian(q[i, :], p[i, :], par)

    if i >= thr:
        mask = energies_i > 0
        q_mask[i-thr] = q[i, :][mask]
        p_mask[i-thr] = p[i, :][mask]
        p_vars[i-thr] = np.var(p_mask[i-thr]) 

times_list = np.linspace(0, par.t-par.dt, p_vars.shape[0])

plt.scatter(times_list, p_vars, s=2)
plt.xlabel("Time [s]")
plt.ylabel("Variance")
plt.title(r"$<\delta ^2>$ vs Time for FCC-ee")


#%%

n_times = q.shape[0]
#n_times = 50
steps = int(len(times)/n_times)


#%%

# DA USARE PER LE TRAIETTORIE DELL'ISOLA

import alphashape
import shapely

x = np.zeros((n_times, n_particles))
y = np.zeros((n_times, n_particles))

actions_list = np.zeros((n_times, n_particles))

for j in tqdm(range(n_particles)):
    for i in range(n_times):
        h_0 = fn.H0_for_action_angle(q[i, j], p[i, j], par)
        kappa_squared = 0.5 * (1 + h_0 / (par.A**2))

        if 0 < kappa_squared < 1:
            Q = (q[i, j] + np.pi) / par.lambd
            P = par.lambd * p[i, j]

            action, theta = fn.compute_action_angle(kappa_squared, P)
            actions_list[i, j] = action 

            x[i, j] = np.sqrt(2 * action) * np.cos(theta)
            y[i, j] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(q[i, j]-np.pi)

x = np.array(x)
y = np.array(y)

areas = []

for j in tqdm(range(n_particles)):
    xy_j = np.vstack((x[:, j], y[:, j])).T
    
    hull_j = alphashape.alphashape(xy_j, alpha=0.4)

    areas.append(hull_j.area)
    if hull_j is not None:
        if hasattr(hull_j, "geoms"):  # MultiPolygon
            for geom in hull_j.geoms:
                x_hull, y_hull = geom.exterior.xy
                plt.plot(x_hull, y_hull, c="r", label="Hull")
        else:  # Polygon
            x_hull, y_hull = hull_j.exterior.xy
            plt.plot(x_hull, y_hull, c="r", label="Hull")
    plt.title(f"Alpha shape hull - particella {j}")
    plt.legend()
    plt.show()


#%%

# DA USARE PER LE TRAIETTORIE DEL CENTRO

actions = np.empty((n_times, n_particles))
energies = []
kappas = []

for i, idx in enumerate(range(0, len(times), steps)):
    h_0 = fn.H0_for_action_angle(q[idx, :], p[idx, :], par)
    kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
    kappas.append(kappa_squared)

    actions[i, :], _ = fn.compute_action_angle(kappa_squared, 1)
    
    h0_of_I = 2 * par.A**2 * (kappa_squared - 1/2)
    energies.append(h0_of_I)
    energies.append(h_0)

E0 = fn.hamiltonian(np.pi, 0, par)
energies = [e_i[e_i > 0] for e_i in energies]

#%%

# TEMPERATURA

noise_D = (par.gamma / par.beta**2 * np.sqrt(2 * par.damp_rate * par.h * par.eta * par.Cq / par.radius))**2
damping_factor = 2 * par.damp_rate / par.beta**2
temperature = noise_D / (2 * damping_factor)


#%%

from scipy.special import rel_entr

chi2_tot = []

#actions = areas / (2 * np.pi)

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
    hist, bin_edges = np.histogram(energies_i, bins=100, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # curva teorica
    P_H = np.exp(- (np.interp(bin_centers, actions_sorted_i, energies_i) - np.min(energies) / (par.k_lele_fcc * temperature)))
    Z = trapezoid(P_H, bin_centers)           
    P_H /= Z 

    epsilon = 1e-16

    chi2 = np.sum((hist - P_H)**2 / (P_H + epsilon))
    chi2_tot.append(chi2)

    plt.hist(energies_i, bins=100, density=True, alpha=0.5, label="Empirico")
    plt.plot(bin_centers, P_H, label="Teorico")
    plt.legend()
    plt.show()

plt.hist(actions, bins=100, density=True, alpha=0.5, label="Distr. of actions")
plt.hist(actions[-1, :], bins=100, density=True, alpha=0.5, label="Distr. of actions")
plt.plot(bin_centers, P_H, label="Boltz. distribution")
plt.title(rf"$\chi^2$: {chi2:.2f}")
plt.legend()
plt.xlabel("Actions")
plt.ylabel("Frequency")
plt.show()
