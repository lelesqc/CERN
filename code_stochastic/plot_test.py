import numpy as np
import matplotlib.pyplot as plt
import params as par
import os
import functions as fn
from tqdm.auto import tqdm

def plot_test():
    data = np.load("stochastic_studies/adiab_invariant/vars_and_avg_energies.npz")

    vars = data["vars"]
    energies = data["energies"]
    mean = []

    times = np.linspace(0, par.T_tot, len(energies))

    for i in range(len(energies)):
        mean.append(np.mean(energies[i, :]))

    plt.scatter(times, mean, s=1) 
    plt.show()

def altro():
    folder = "./stochastic_studies/adiab_invariant/ang_coeff_vs_exc_amplitude"
    epsilon_f_list = []
    slope_list = []

    for fname in os.listdir(folder):
        if fname.endswith("fcc.npz"):
            data = np.load(os.path.join(folder, fname))
            epsilon_f_list.append(data["epsilon_f"].item() / par.nu_m_f)
            slope_list.append(data["slope"].item())

    plt.scatter(epsilon_f_list[:-1], slope_list[:-1], s=20)
    plt.xlabel("Modulation amplitude")
    plt.ylabel("Slope")
    plt.yscale("log")
    #plt.savefig("../results/resonance11/center/slope_vs_a_mean_energy_fcc.png")
    plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
import params as par
import os
import functions as fn
from tqdm.auto import tqdm

data = np.load("integrator/evolved_qp_all_middle.npz")
q = data["q"][::50, :]
p = data["p"][::50, :]
times = data["t_list"]
psi_list = data["psi"]

traj_q = np.zeros((q.shape[0], 100, 10000), dtype=np.float16)
traj_p = np.zeros((q.shape[0], 100, 10000), dtype=np.float16)

plt.scatter(q, p)
plt.show


for i in tqdm(range(q.shape[0])):
    psi_now = psi_list[i]
    t_now = times[i]

    j=0

    q_temp = q[i, :]
    p_temp = p[i, :]

    while j < 100:
        q_temp, p_temp = fn.integrator_step(q_temp, p_temp, psi_now, t_now, par.dt, fn.Delta_q, fn.dV_dq)

        if np.cos(psi_now) > 1.0 - 1e-3:
            traj_q[i, j, :] = np.copy(q_temp)
            traj_p[i, j, :] = np.copy(p_temp)

            j += 1

        psi_now += par.omega_lambda(t_now) * par.dt

#%%

x = np.zeros((traj_q.shape[0], traj_q.shape[1], traj_q.shape[2]))
y = np.zeros((traj_q.shape[0], traj_q.shape[1], traj_q.shape[2]))

for j in tqdm(range(0, x.shape[2], 100)):
    for i in range(x.shape[0]):
        for k in range(x.shape[1]):
            h_0 = fn.H0_for_action_angle(traj_q[i, k, j], traj_p[i, k, j])
            kappa_squared = 0.5 * (1 + h_0 / (par.A**2))

            if 0 < kappa_squared < 1:
                Q = (traj_q[i, k, j] + np.pi) / par.lambd
                P = par.lambd * traj_p[i, k, j]

                action, theta = fn.compute_action_angle(kappa_squared, P)

                x[i, k, j] = np.sqrt(2 * action) * np.cos(theta)
                y[i, k, j] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(traj_q[i, k, j]-np.pi)

x = np.array(x)
y = np.array(y)


#%%

plt.scatter(x[10, :, 10], y[10, :, 10])
plt.show()

#%%

if __name__ == "__main__":
    #plot_test()
    #altro_ancora()

