#%%

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import params_fcc  as par
import functions as fn

integrator_data = np.load("./integrator/evolved_qp_evolution.npz")
phasespace = np.load("../phasespace_code/integrator/evolved_qp_phasespace.npz")
phasespace_xy = np.load("../phasespace_code/action_angle/phasespace_a0.050_nu0.80.npz")

q = integrator_data["q"]
p = integrator_data["p"]
psi = integrator_data["psi"]
time = integrator_data["time"]

q_ps = phasespace["q"]
p_ps = phasespace["p"]

x_ps = phasespace_xy["x"]
y_ps = phasespace_xy["y"]

q_out = q.copy()
p_out = p.copy()

q = q_out
p = p_out

step = 0
steps = 500
psi = psi
par.t = time

#par.a = par.epsilon / par.nu_m

while step < steps:
    q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

    if np.cos(psi) > 1.0 - 1e-3:
        q_last = q
        p_last = p

        step += 1

    psi += par.omega_m * par.dt
    par.t += par.dt

q = q_last
p = p_last

#%%

energies = np.abs(fn.hamiltonian(q, p))
temperature = np.mean(energies)

print(temperature)


#%%

if q.ndim == 1:
        n_steps = 1
        n_particles = q.shape[0]
        q = q.reshape((1, n_particles))
        p = p.reshape((1, n_particles))
else:
    n_steps, n_particles = q.shape

actions_list = np.zeros((n_steps, n_particles))

x = np.zeros((n_steps, n_particles))
y = np.zeros((n_steps, n_particles))

for j in tqdm(range(n_particles)):
    for i in range(n_steps):
        h_0 = fn.H0_for_action_angle(q[i, j], p[i, j])
        kappa_squared = 0.5 * (1 + h_0 / (par.A**2))

        if 0 < kappa_squared < 1:
            Q = (q[i, j] + np.pi) / par.lambd
            P = par.lambd * p[i, j]

            action, theta = fn.compute_action_angle(kappa_squared, P)
            actions_list[i, j] = action 

            x[i, j] = np.sqrt(2 * action) * np.cos(theta)
            y[i, j] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(q[i, j]-np.pi)

x_final = np.array(x)
y_final = np.array(y)

#%%

energies = np.array(fn.hamiltonian(q[0, :], p[0, :]))

mask = energies < 0
energies_center = energies[mask]
energies_island = energies[~mask]

q_island = q[0, :][~mask]
p_island = p[0, :][~mask]
q_mean_isl = np.mean(q_island)
p_mean_isl = np.mean(p_island)

q_center = q[0, :][mask]
p_center = p[0, :][mask]
q_mean_cen = np.mean(q_center)
p_mean_cen = np.mean(p_center)

x_island = x_final[0, :][~mask]
y_island = y_final[0, :][~mask]

h0_island_mean = fn.hamiltonian(q_mean_isl, p_mean_isl)

#plt.scatter(x_ps, y_ps, s = 1)
#plt.scatter(x_final, y_final, s = 1)

plt.hist(energies_island - h0_island_mean, bins=100)
#plt.hist(energies_center, bins=100)
plt.show()
# %%

energies = fn.hamiltonian(q_island, p_island)

plt.figure(figsize=(8, 6))
plt.scatter(x_ps, y_ps, s = 0.5)
sc = plt.scatter(x_island, y_island, c=energies, cmap='viridis', s=1.5)
plt.colorbar(sc, label='Energia finale')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Traiettorie colorate per energia finale')
plt.show()

# %%
