import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import params
import functions as fn

def run_integrator(poincare_mode, idx_start, idx_end, params_path):
    par = params.load_params(params_path)
    data = np.load(f"init_conditions/init_distribution_{idx_start}_{idx_end}.npz")
    fn.par = par
    #data_evolved = np.load("integrator/evolved_qp_last_relaxed_fcc.npz")
    #time = data_evolved["t_final"]
    #psi = data_evolved["psi"]

    q_init = data['q']
    p_init = data['p']

    q = np.copy(q_init)
    p = np.copy(p_init)

    q_single = None
    p_single = None

    if poincare_mode != "last":
        q_sec = np.empty((par.n_steps + 1, *q.shape), dtype=np.float16)
        p_sec = np.empty((par.n_steps + 1, *p.shape), dtype=np.float16)

    sec_count = 0
    avg_energies = []
    vars = []

    step = 0
    psi = par.phi_0
    find_poincare = False
    fixed_params = False

    psi_list = []
    times_list = []

    while not find_poincare:
        if par.t >= par.T_tot:
            fixed_params = True

        q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

        if np.cos(psi) > 1.0 - 1e-3:
            if poincare_mode == "all":
                q_sec[sec_count, :] = np.copy(q)
                p_sec[sec_count, :] = np.copy(p)
                sec_count += 1

                psi_list.append(psi)
                times_list.append(par.t)

                psi_final = psi
                t_final = par.t

                if fixed_params:
                    find_poincare = True
                    
                    break

            elif poincare_mode == "last" and fixed_params:
                q_single = np.copy(q)
                p_single = np.copy(p)
                find_poincare = True
                psi_final = psi
                t_final = par.t

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
        q = q_sec[:sec_count, :]
        p = p_sec[:sec_count, :]
    else:
        q = q_single
        p = p_single

    q = np.array(q)
    p = np.array(p)

    #np.savez("./init_conditions/relaxed_qp_als.npz", q=q, p=p)
    
    return q, p, psi_list, t_final


# --------------- Save results ----------------


if __name__ == "__main__":
    poincare_mode = sys.argv[1]
    idx_start = int(sys.argv[2])
    idx_end = int(sys.argv[3])
    params_path = sys.argv[4] if len(sys.argv) > 4 else "params.yaml"
    q, p, psi, t_list = run_integrator(poincare_mode, idx_start, idx_end, params_path)

    output_dir = "integrator"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"evolved_qp_{poincare_mode}_{idx_start}_{idx_end}.npz")
    np.savez(file_path, q=q, p=p, psi=psi, t_list=t_list)