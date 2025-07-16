import os
import sys
import numpy as np

import params as par
import functions as fn

def run_integrator(mode):
    data = np.load("init_conditions/qp.npz")

    q_init = data['q']
    p_init = data['p']

    q = np.copy(q_init)
    p = np.copy(p_init)

    q_last = np.copy(q_init)
    p_last = np.copy(p_init)

    if mode == "tune":
        q_traj = np.zeros((par.n_steps, len(q)))
        p_traj = np.zeros((par.n_steps, len(p)))
        step_count = 0
        
    elif mode == "phasespace":
        #q_sec = np.zeros((par.n_steps, len(q)))
        #p_sec = np.zeros((par.n_steps, len(p)))
        sec_count = 0

    psi = par.phi_0

    for step in range(par.n_steps):
        q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

        if mode == "tune":
            q_traj[step_count] = q
            p_traj[step_count] = p
            step_count += 1

        elif mode == "phasespace":
            if np.cos(psi) > 1.0 - 1e-3: 
                q_last = q
                p_last = p             
                #q_sec[sec_count] = q
                #p_sec[sec_count] = p
                sec_count += 1
                
        psi += par.omega_m * par.dt
        par.t += par.dt

    if mode == "tune":
        q = q_traj
        p = p_traj

    elif mode == "phasespace":
        #q = q_sec[:sec_count]
        #p = p_sec[:sec_count]
        q = q_last
        p = p_last

    return q, p


# --------------- Save results ----------------


if __name__ == "__main__":
    mode = sys.argv[1]
    q, p = run_integrator(mode)

    output_dir = "integrator"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"evolved_qp_{mode}.npz")
    np.savez(file_path, q=q, p=p)