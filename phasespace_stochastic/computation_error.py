#%%

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.integrate import trapezoid

import params_fcc
import functions as fn

import warnings
warnings.filterwarnings("ignore")

par = params_fcc.Params()

init_data = np.load("./init_conditions/init_qp_10000_gaussian_fcc.npz")

q = init_data["q"]
p = init_data["p"]

psi = par.phi_0
par.t = 0

n_particles = int(q.shape[0])
n_times = int(q.shape[0])

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

steps = int(len(times)/n_times)


#%%

# actions for particles in the resonant island

import alphashape

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

# actions for particles in the center

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


#%%

from scipy.stats import linregress


L2_list = []
dof = 98

E0 = fn.hamiltonian(np.mean(q), np.mean(p), par)

for i in range(n_times):
    nan_indices = np.where(np.isnan(actions[i, :]))[0]
    if nan_indices.size > 0:
        print(f"NaN in actions[{i},:] at indices: {nan_indices}")
        print("Correspondent values of q:", q[i, :][nan_indices])
        print("Correspondent values of p:", p[i, :][nan_indices])
        print(kappas[i][nan_indices])
    energies_i = energies[i]
    sorted_idx = np.argsort(actions[i, :])
    energies_i = energies_i[sorted_idx]
    actions_sorted_i = actions[i, :][sorted_idx]
   
    # histogram
    hist, bin_edges = np.histogram(actions[i, :], bins=100, density=False)
    P_continuous = np.exp(-(energies_i - E0) / par.temperature)
    Z = trapezoid(P_continuous, actions_sorted_i)
    P_continuous /= Z

    P_H_bin = np.zeros_like(hist)
    for j in range(len(hist)):
        x0, x1 = bin_edges[j], bin_edges[j+1]
        mask = (actions_sorted_i >= x0) & (actions_sorted_i < x1)
        if np.any(mask):
            P_H_bin[j] = trapezoid(P_continuous[mask], actions_sorted_i[mask]) / (x1 - x0)
        else:
            P_H_bin[j] = np.interp(0.5*(x0+x1), actions_sorted_i, P_continuous)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)  # width of each bin (const)
    L2 = np.sqrt(trapezoid((hist - P_H_bin)**2, bin_centers))
    L2_theory = np.sqrt(trapezoid(P_H_bin**2))

    epsilon = 1e-15
    L2_rel = L2 / L2_theory
    
    L2_list.append(L2_rel)

    chi2_tot = []

plt.hist(actions[-1, :], bins=100, density=True, alpha=0.5, label="Distr. of actions")
plt.plot(bin_centers, P_H_bin, label="Boltz. distribution")
plt.title(f"L2 norm: {L2_list[-1]:.4f}")
plt.legend()
plt.xlabel("I")
plt.ylabel(r"$\rho(I)$")
plt.show()

plt.hist(actions[0, :], bins=100, density=True, alpha=0.5, label="Distr. of actions")
plt.plot(bin_centers, P_H_bin, label="Boltz. distribution")
plt.title(f"L2 norm: {L2_list[0]:.4f}")
plt.legend()
plt.xlabel("I")
plt.ylabel(r"$\rho(I)$")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)

fontsize_title = 34
fontsize_label = 38
fontsize_tick = 28
fontsize_legend = 32

# initial actions
axes[0].hist(actions[0, :], bins=100, density=True, alpha=0.5, label=r"$\rho(I)$")
axes[0].plot(bin_centers, P_H_bin, label=r"$\rho_\text{MB}$", lw=3.5)
axes[0].set_title(rf"t = 0s", fontsize=fontsize_title)
axes[0].set_xlabel("I", fontsize=fontsize_label)
axes[0].set_ylabel(r"$\rho(I)$", fontsize=fontsize_label)
axes[0].set_xlim(0, 0.01)
axes[0].set_ylim(0, 1000)
axes[0].tick_params(axis='both', labelsize=fontsize_tick)
axes[0].legend(fontsize=fontsize_legend)

# final actions
axes[1].hist(actions[-1, :], bins=100, density=True, alpha=0.5, label=r"$\rho(I)$")
axes[1].plot(bin_centers, P_H_bin, label=r"$\rho_\text{MB}$", lw=3.5)
axes[1].set_title(rf"t = 4.2s", fontsize=fontsize_title)
axes[1].set_xlabel("I", fontsize=fontsize_label)
axes[1].set_xlim(0, 0.01)
axes[1].set_ylim(0, 1000)
axes[1].tick_params(axis='both', labelsize=fontsize_tick)
axes[1].legend(fontsize=fontsize_legend)

plt.show()

#%%

timez = np.linspace(0, times, len(L2_list))

timez = np.asarray(timez)
L2_arr = np.asarray(L2_list)
mask = np.isfinite(timez) & np.isfinite(L2_arr) & (L2_arr > 0)

tmin, tmax = 0.0, 0.05
if tmin is not None or tmax is not None:
    mwin = np.ones_like(timez, dtype=bool)
    if tmin is not None:
        mwin &= timez >= tmin
    if tmax is not None:
        mwin &= timez <= tmax
    mask &= mwin

x = timez[mask]
y = np.log(L2_arr[mask])

# Linear fit: ln(L2) = a + b * t
res = linregress(x, y)
a, b = res.intercept, res.slope  # slope b [1/s] log-scale
r2 = res.rvalue**2
tau = (-1.0 / b) if b < 0 else np.inf  # decay constant
sigma_b = res.stderr
if b != 0:
    sigma_tau = abs(sigma_b / b**2)
else:
    sigma_tau = np.nan

print(f"Fit, tau={tau:.3f} s, damping time: {1 / (2 * par.damp_rate)}, damping rate: {par.damp_rate}, U0: {par.U_0}")

x_fit = np.linspace(x.min(), x.max(), 200)
L2_fit = np.exp(a + b * x_fit)

fig, ax = plt.subplots(figsize=(22, 12), constrained_layout=True)
ax.scatter(timez, L2_list, s=14, label="data")
ax.plot(x_fit, L2_fit, 'r-', lw=5, label=rf"$\tau_\text{{rel}}$=({tau:.4f} $\pm$ {sigma_tau:.4f})s, $T_\text{{tot}} = {times[-1]:.1f}s$")
ax.set_xlabel("Time [s]", fontsize=44)
ax.set_yscale("log")
ax.set_ylabel(r"$|| \rho(I) - \rho_\text{MB} ||_2$", fontsize=44)
ax.legend(fontsize=38)
ax.tick_params(axis='both', labelsize=38)
plt.subplots_adjust(left=0.3)
plt.tight_layout(pad=3.0)
plt.show()
