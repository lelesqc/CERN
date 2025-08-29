import os
import yaml
import random
import numpy as np
from tqdm import tqdm

import params as par
import functions as fn
import generate_init_htcondor as gen_init
import matplotlib.pyplot as plt

def main_script(radius, n_particles, seed):
    base_dir = os.environ.get("BASE_DIR", "/mnt/c/Users/emanu/OneDrive - Alma Mater Studiorum Università di Bologna/CERN_data/code")

    """
    q, p = gen_init.generate_init(radius, n_particles, seed)

    psi = 0
    par.t = 0
    poincare_points = 1000

    q_ext = np.zeros((par.n_steps, n_particles))
    p_ext = np.zeros((par.n_steps, n_particles))
    sec_count = 0

    times = []
    psi_vals = []

    while par.t < par.T_tot: 
        q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

        if np.cos(psi) > 1.0 - 1e-3:              
            q_ext[sec_count] = np.copy(q)
            p_ext[sec_count] = np.copy(p)
            times.append(par.t)
            psi_vals.append(psi)
            sec_count += 1
            if sec_count == par.n_steps // 2:
                print("half-way")

        psi += par.omega_lambda(par.t) * par.dt
        par.t += par.dt

    q = q_ext[:sec_count, :]    # (poincare_found, n_particles)
    p = p_ext[:sec_count, :]

    np.savez(base_dir + "/actions_stuff/particle_data_phasespace.npz", q=q, p=p, times=times, psi_vals=psi_vals) 

    inner_traj_q = np.zeros((n_particles, q.shape[0], poincare_points), dtype=np.float16)    # i-th particle, j-th ext step, k-th int step
    inner_traj_p = np.zeros((n_particles, p.shape[0], poincare_points), dtype=np.float16)

    psi_tracker = psi_vals

    tunes_total = np.zeros((q.shape[0], n_particles), dtype=np.float16)

    for j in tqdm(range(q.shape[0])):
        q_temp = q[j, :]
        p_temp = p[j, :]
        psi_temp = psi_tracker[j]

        tunes = np.zeros(n_particles)
        prev_angles = np.zeros(n_particles)
        counts = np.zeros(n_particles, dtype=int)

        z0 = (q_temp - np.mean(q_temp)) - 1j * p_temp
        angle_prev = np.angle(z0)
        angle_unwrapped_prev = angle_prev    # si parte dall'angolo iniziale
        tune_curr = 0.0
        count_curr = 0

        int_sec_count = 0

        while int_sec_count < poincare_points:
            q_temp, p_temp = fn.integrator_step_fixed(q_temp, p_temp, psi_temp, par.a_lambda(times[j]), par.omega_lambda(times[j]), par.dt, fn.Delta_q_fixed, fn.dV_dq)
            
            z_curr = (q_temp - np.mean(q_temp)) - 1j * p_temp
            angle_curr = np.angle(z_curr)

            tune_curr, count_curr, angle_unwrapped_prev = fn.avg_phase_adv_runtime(angle_unwrapped_prev, angle_curr, tune_curr, count_curr)
            angle_prev = angle_curr

            if np.cos(psi_temp) > 1.0 - 1e-3:
                inner_traj_q[:, j, int_sec_count] = q_temp
                inner_traj_p[:, j, int_sec_count] = p_temp

                int_sec_count += 1

            psi_temp += par.omega_lambda(times[j]) * par.dt

            tunes = tune_curr

        tunes_total[j, :] = tunes

    np.savez(base_dir + "/integrator/phasespaces_qp.npz", q=inner_traj_q, p=inner_traj_p) 
    np.savez(base_dir + "/integrator/tunes_phasespaces.npz", tunes = tunes_total)
    #data_qp = np.load(base_dir + "/integrator/phasespaces_qp.npz")
    #q = data_qp["q"]
    #p = data_qp["p"]

    #print(q.shape)

    """
    data_coord = np.load(base_dir + "/integrator/phasespaces_qp.npz")
    data_tune = np.load(base_dir + "/integrator/tunes_phasespaces.npz")

    inner_traj_q = data_coord["q"]
    inner_traj_p = data_coord["p"]
    tunes = data_tune["tunes"]

    idx_list = np.round(np.linspace(0, inner_traj_q.shape[1] - 1, 20)).astype(int)

    x = np.zeros((inner_traj_q.shape[0], inner_traj_q.shape[1], inner_traj_q.shape[2]), dtype=np.float16)
    y = np.zeros((inner_traj_p.shape[0], inner_traj_p.shape[1], inner_traj_p.shape[2]), dtype=np.float16)

    for i in tqdm(range(x.shape[0])):   # ciclo sulle particelle
        #for j in range(x.shape[1]):     # ciclo sui punti esterni
        for j in idx_list:
            for k in range(x.shape[2]):    # ciclo sui punti interni
                h_0 = fn.H0_for_action_angle(inner_traj_q[i, j, k], inner_traj_p[i, j, k])
                kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
                if 0 < kappa_squared < 1:
                    #Q = (inner_traj_q[i, j, k] + np.pi) / par.lambd
                    P = par.lambd * inner_traj_p[i, j, k]
                    action, theta = fn.compute_action_angle(kappa_squared, P)
                    x[i, j, k] = np.sqrt(2 * action) * np.cos(theta)
                    y[i, j, k] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(inner_traj_q[i, j, k] - np.pi)

    return x, y

if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    radius = float(params["radius"])
    n_particles = int(params["particles"])
    seed = int(params.get("seed", 42))

    mypath = "."

    np.random.seed(seed)
    random.seed(seed)

    x, y = main_script(radius, n_particles, seed)

    output_dir = "./output_ps" 
    outname = f"results_seed{seed}_test.npz"

    os.makedirs(output_dir, exist_ok=True) 
    file_path = os.path.join(output_dir, outname)
    np.savez(file_path, x=x, y=y)

    print(f"Saved results to {outname}")