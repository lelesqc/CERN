#%%

import numpy as np
import matplotlib.pyplot as plt
import params_fcc as par
import functions as fn
import alphashape
from tqdm.auto import tqdm

a_start = par.a_lambda(par.T_percent)
omega_start = par.omega_lambda(par.T_percent)
a_end = par.a_lambda(par.T_tot)
omega_end = par.omega_lambda(par.T_tot)

a_start_str = f"{a_start:.3f}"
omega_start_str = f"{omega_start:.7f}"
a_end_str = f"{a_end:.3f}"
omega_end_str = f"{omega_end:.7f}"

str_title = f"a{a_start_str}-{a_end_str}_nu{float(omega_start_str)/par.omega_s:.7f}-{float(omega_end_str)/par.omega_s:.3f}"


final_data = np.load(f"../phasespace_stochastic/integrator/evolution_qp_10000_fcc.npz")
#final_data = np.load("./integrator/evolved_qp_last_0_10000.npz")
#data_xy = np.load(f"./action_angle/last_{str_title}_0_10000.npz")
data_xy = np.load(f"../phasespace_stochastic/action_angle/evolution_10000_a{a_end:.7f}_nu0.83000_fcc.npz")

jump = 1


q = final_data["q"][::jump]
p = final_data["p"][::jump]
psi = final_data["psi"]
time = final_data["time"]
x_fin = data_xy["x"][0][::jump]
y_fin = data_xy["y"][0][::jump]

print(x_fin.shape)


mask = ((x_fin+0.2)**2 + y_fin**2) > 1.5

par.damp_rate = 0
par.D = 0

steps = 250

q_traj = np.zeros((steps, q.shape[0]))
p_traj = np.zeros((steps, q.shape[0]))

# Salva le condizioni iniziali come primo elemento
q_traj[0, :] = q
p_traj[0, :] = p

a = par.a_lambda(par.T_tot)
omega_m = par.omega_lambda(par.T_tot)

step = 1 
while step < steps:
    q, p = fn.integrator_step_fixed(q, p, psi, a, omega_m, par.dt, fn.Delta_q_fixed, fn.dV_dq)

    if np.cos(psi) > (1.0 - 1e-4):
        q_traj[step, :] = np.copy(q)
        p_traj[step, :] = np.copy(p)

        step += 1

    psi += omega_m * par.dt

x_traj = np.zeros((steps, q_traj.shape[1]))
y_traj = np.zeros((steps, q_traj.shape[1]))

actions_var = np.zeros((steps, q_traj.shape[1]))
par.t = time

energies = []

for j in tqdm(range(x_traj.shape[1])):
    for i in range(x_traj.shape[0]):
        if i == 0:
            energy = fn.hamiltonian(q_traj[i, j], p_traj[i, j])
            energies.append(energy)

        h_0 = fn.H0_for_action_angle(q_traj[i, j], p_traj[i, j])
        kappa_squared = 0.5 * (1 + h_0 / (par.A**2))

        if 0 < kappa_squared < 1:
            Q = (q_traj[i, j] + np.pi) / par.lambd
            P = par.lambd * p_traj[i, j]

            action, theta = fn.compute_action_angle(kappa_squared, P)

            actions_var[i, j] = action
        
            x_traj[i, j] = np.real(np.sqrt(2 * action) * np.cos(theta))
            y_traj[i, j] = np.real(-np.sqrt(2 * action) * np.sin(theta) * np.sign(q_traj[i, j]-np.pi))


#x_traj = x_traj[:, mask] if np.any(mask) else np.empty((steps, 0))
#y_traj = y_traj[:, mask] if np.any(mask) else np.empty((steps, 0))
#actions_cen = actions_var[0, ~mask] if np.any(~mask) else np.empty((0,))

actions_isl = []
actions_cen = []

energies_cen = []

for k in range(x_traj.shape[1]):
    xy = np.vstack((x_traj[:, k], y_traj[:, k])).T
    alpha = 0.4
    hull = alphashape.alphashape(xy, alpha)
    area = getattr(hull, "area", 0.0) if hull is not None else 0.0
    action_final = area / (2 * np.pi)
    if mask[k]:
        actions_isl.append(action_final)
    else:
        actions_cen.append(action_final)
        energies_cen.append(energies[k])

    


    """plt.figure()
    plt.scatter(xy[:, 0], xy[:, 1], s=5, c='r', label="Punti")
    if hull is not None and not hull.is_empty:
        try:
            x_hull, y_hull = hull.exterior.xy
            plt.plot(x_hull, y_hull, c='b', label="Hull alphashape")
        except AttributeError:
            for geom in hull.geoms:
                x_hull, y_hull = geom.exterior.xy
                plt.plot(x_hull, y_hull, c='b', label="Hull alphashape")
    plt.title(f"Hull particella {k}, area={area:.3f}")
    plt.legend()
    plt.show()"""

#np.savez(f"./study_actions/nu{float(omega_start_str)/par.omega_s:.4f}_actions.npz", energies=energies_cen, actions_isl=np.array(actions_isl), actions_cen=actions_cen, nu_i=float(omega_start_str)/par.omega_s)

# %%

print(actions_cen[5])

#%%

plt.hist(actions_cen, bins=int(np.round(np.sqrt(len(actions_cen)))))
plt.show()

print(np.mean(actions_cen))


# %%
