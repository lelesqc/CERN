import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import params as par
import functions as fn

base_dir = os.environ["BASE_DIR"]

def run_integrator(poincare_mode, n_particles):
    """
    Run the symplectic integrator to evolve the system.

    """
    data = np.load(base_dir + "/init_conditions/init_distribution.npz")

    q_init = data['q']
    p_init = data['p']

    q = np.copy(q_init)
    p = np.copy(p_init)

    q_single = None
    p_single = None

    if poincare_mode == "none":
        q_all = np.empty((par.n_steps // 7, n_particles))
        p_all = np.empty((par.n_steps // 7, n_particles))
        q_all[0, :] = np.copy(q)
        p_all[0, :] = np.copy(p)

    if poincare_mode == "all":
        q_sec = np.empty((par.n_steps, n_particles))
        p_sec = np.empty((par.n_steps, n_particles))
        sec_count = 0

    step = 0
    par.t = 0
    psi = par.phi_0
    find_poincare = False
    fixed_params = False
    
    tunes = np.zeros(n_particles)
    prev_angles = np.zeros(n_particles)
    counts = np.zeros(n_particles, dtype=int)

    z0 = (q - np.mean(q)) - 1j * p
    angle_prev = np.angle(z0)
    angle_unwrapped_prev = angle_prev    # si parte dall'angolo iniziale
    tune_curr = 0.0
    count_curr = 0

    while not find_poincare:
        if par.t >= par.T_tot:
            fixed_params = True

        q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)   

        if poincare_mode == "none":
            if step < q_all.shape[0]:  
                q_all[step+1, :] = np.copy(q)
                p_all[step+1, :] = np.copy(p)
                if par.t >= par.T_percent:
                    break
            if par.t >= par.T_tot:
                find_poincare = True
                psi_val = psi
                break
        if np.cos(psi) > 1.0 - 1e-3 and poincare_mode != "none":
            if poincare_mode == "first":
                if q_single is None:
                    q_single = np.copy(q)
                    p_single = np.copy(p)
                    find_poincare = True
                    psi_val = psi
                    break
            elif poincare_mode == "all":
                q_sec[sec_count] = np.copy(q)
                p_sec[sec_count] = np.copy(p)
                sec_count += 1
                if fixed_params:
                    find_poincare = True
                    psi_val = psi
                    break
            elif poincare_mode == "last" and fixed_params:
                q_single = np.copy(q)
                p_single = np.copy(p)
                find_poincare = True
                psi_val = psi
                break
            
        psi += par.omega_lambda(par.t) * par.dt
        par.t += par.dt
        step += 1

        if step == par.n_steps // 4:
            print(r">>> 25% completed")
        elif step == par.n_steps // 2:
            print(r">>> 50% completed")
        elif step == 3 * par.n_steps // 4:
            print(r">>> 75% completed")       

    if poincare_mode == "all":
        q = q_sec[:sec_count]
        p = p_sec[:sec_count]
    elif poincare_mode == "none":
        q = np.array(q_all[:step])
        p = np.array(p_all[:step])
    else:
        q = q_single
        p = p_single

    q_temp = q.copy()
    p_temp = p.copy()
    psi_temp = psi

    print(par.omega_lambda(par.t))

    extra_steps = 16384
    if poincare_mode == "last":
        for _ in tqdm(range(extra_steps)):
            q_temp, p_temp = fn.integrator_step(q_temp, p_temp, psi_temp, par.t, par.dt, fn.Delta_q, fn.dV_dq)  

            z_curr = (q_temp - np.mean(q_temp)) - 1j * p_temp
            angle_curr = np.angle(z_curr)

            tune_curr, count_curr, angle_unwrapped_prev = fn.avg_phase_adv_runtime(angle_unwrapped_prev, angle_curr, tune_curr, count_curr)
            angle_prev = angle_curr

            psi_temp += par.omega_lambda(par.t) * par.dt

            tunes = tune_curr

    print(par.omega_lambda(par.t))
    q = np.array(q)
    p = np.array(p)

    psi_val = psi

    return q, p, psi_val, tunes


# --------------- Save results ----------------


if __name__ == "__main__":
    poincare_mode = sys.argv[1]
    n_particles = int(sys.argv[2])
    
    q, p, psi_val, tunes = run_integrator(poincare_mode, n_particles)

    output_dir = base_dir + "/integrator"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"evolved_qp_{poincare_mode}_{n_particles}.npz")
    np.savez(file_path, q=q, p=p, psi=psi_val, tunes=tunes)