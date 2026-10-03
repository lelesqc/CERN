import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import params_fcc as par
import functions as fn

def run_integrator(poincare_mode, idx_start, idx_end):
    data = np.load(f"init_conditions/init_distribution_{idx_start}_{idx_end}.npz")

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
        q_sec = np.empty((par.n_steps + 1, q_init.shape[0]), dtype=np.float16)
        p_sec = np.empty((par.n_steps + 1, q_init.shape[0]), dtype=np.float16)
    
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

        if np.cos(psi) > (1.0 - 1e-3):
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
            #elif poincare_mode == "last" and par.t >= par.T_percent: 
                q_single = np.copy(q)
                p_single = np.copy(p)
                find_poincare = True
                psi_final = psi
                t_final = par.t

                #print(par.omega_lambda(t_final) / par.omega_s, par.epsilon_lambda(t_final))

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

        #mask = (times_list >= par.T_percent) & (times_list < (par.T_tot / 6))
        mask = times_list <= par.T_percent

        #print(par.T_percent, par.T_tot)        

        q = np.copy(q[mask, :])
        p = np.copy(p[mask, :])
        times_list = np.array(times_list)
        times_list = times_list[mask]

        # Seleziona 10 indici equispaziati tra il primo e l'ultimo
        num_points = 100
        indices = np.linspace(0, q.shape[0] - 1, num_points, dtype=int)

        q = q[indices, :]
        p = p[indices, :]
        times_list = times_list[indices]
        times_list = np.array(times_list)

        #np.savez("../phasespace_stochastic/params_for_gif_add.npz", t_list=times_list, eps_list=par.epsilon_lambda(times_list), nu_list=par.omega_lambda(times_list)/par.omega_s)

        #indices = np.linspace(0, q.shape[0] - 1, 10, dtype=int)
        #q = np.copy(q[indices])
        #p = np.copy(p[indices])

        #times_list = np.array(times_list)
        #times_list = times_list[indices]

        
        #np.savez("./times/times_als_lasttt_lasciastare_add.npz", times_list=times_list)

        #plt.scatter(q[0, :], p[0, :], s=1)
        #plt.show()
        #plt.scatter(q[-1, :], p[-1, :], s=1)
        #plt.show()

        print(len(times_list))
        print(times_list)

    else:
        q = q_single
        p = p_single

    q = np.array(q)
    p = np.array(p)
    
    #np.savez("./init_conditions/relaxed_qp_als.npz", q=q, p=p)
    
    return q, p, psi_final, t_final


# --------------- Save results ----------------


if __name__ == "__main__":
    poincare_mode = sys.argv[1]
    idx_start = int(sys.argv[2])
    idx_end = int(sys.argv[3])
    q, p, psi, t_list = run_integrator(poincare_mode, idx_start, idx_end)

    output_dir = "integrator"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"evolved_qp_{poincare_mode}_{idx_start}_{idx_end}.npz")
    np.savez(file_path, q=q, p=p, psi=psi, t_list=t_list)