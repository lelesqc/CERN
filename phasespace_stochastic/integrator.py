import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import params_fcc
import functions as fn

par = params_fcc.Params()

def run_integrator(mode, n_particles):
    data = np.load(f"init_conditions/qp_{n_particles}.npz")

    q_init = data['q']
    p_init = data['p']

    q = np.copy(q_init)
    p = np.copy(p_init)

    psi = par.phi_0

    quarter = int(par.n_steps * 0.25)
    half = int(par.n_steps * 0.5)
    three_quarters = int(par.n_steps * 0.75)

    if mode == "phasespace":
        q_traj = np.zeros((par.n_steps, len(q)))
        p_traj = np.zeros((par.n_steps, len(p)))
    
    step_count = 0  
    while step_count < par.n_steps:
        q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq, par)

        if mode == "phasespace":
            psi_final=1
            time_final=1
            if np.cos(psi) > 1.0 - 1e-3:
                q_traj[step_count, :] = q
                p_traj[step_count, :] = p

                #print(step_count)

            if step_count == quarter:
                print("un quarto")
            elif step_count == half:
                print("metà") 
            elif step_count == three_quarters:
                print("tre quarti") 
            
            step_count += 1


        elif mode == "evolution":
            if np.cos(psi) > 1.0 - 1e-3: 
                q_last = q
                p_last = p 

                psi_final = psi
                time_final = par.t

            step_count += 1

            if step_count == quarter:
                print("un quarto")
            elif step_count == half:
                print("metà") 
            elif step_count == three_quarters:
                print("tre quarti")  

        psi += par.omega_m * par.dt
        par.t += par.dt     

    if mode == "phasespace":
        q = q_traj
        p = p_traj

    elif mode == "evolution":
        q = np.copy(q_last)
        p = np.copy(p_last)

    return q, p, psi_final, time_final


# --------------- Save results ----------------


if __name__ == "__main__":
    mode = sys.argv[1]
    n_particles = sys.argv[2]
    q, p, psi, time = run_integrator(mode, n_particles)

    output_dir = "integrator"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"evolved_qp_{mode}.npz")
    np.savez(file_path, q=q, p=p, psi=psi, time=time)