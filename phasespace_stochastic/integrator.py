import os
import importlib
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import functions as fn

params_module = os.environ.get("PARAMS_MODULE")
params = importlib.import_module(params_module)
par = params.Params()

machine = os.environ.get("MACHINE").lower()

def run_integrator(init_type, mode, n_particles):
    data = np.load(f"init_conditions/init_qp_{n_particles}_{init_type}_{machine}.npz")

    q_init = data["q"]
    p_init = data["p"]

    q = np.copy(q_init)
    p = np.copy(p_init)

    psi = par.phi_0

    quarter = int(par.n_steps * 0.25)
    half = int(par.n_steps * 0.5)
    three_quarters = int(par.n_steps * 0.75)

    if mode == "phasespace":
        q_traj = np.zeros((par.n_steps, len(q)))
        p_traj = np.zeros((par.n_steps, len(p)))
    
    sec_count = 0
    step_count = 0  
    while step_count < par.n_steps:
        q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq, par)

        if mode == "phasespace":
            psi_final=1
            time_final=1
            if np.cos(psi) > 1.0 - 1e-3:
                q_traj[sec_count, :] = np.copy(q)
                p_traj[sec_count, :] = np.copy(p)
            
                sec_count += 1
                
            if step_count == quarter:
                print("un quarto")
            elif step_count == half:
                print("metà") 
            elif step_count == three_quarters:
                print("tre quarti") 
            
            step_count += 1

        elif mode == "evolution":
            if np.cos(psi) > 1.0 - 1e-3: 
                q_last = np.copy(q)
                p_last = np.copy(p)

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
        q = q_traj[:sec_count, :]
        p = p_traj[:sec_count, :]
        
    elif mode == "evolution":
        q = np.copy(q_last)
        p = np.copy(p_last)

    return q, p, psi_final, time_final


# --------------- Save results ----------------


if __name__ == "__main__":
    machine = os.environ.get("MACHINE").lower()

    init_type = sys.argv[1]
    mode = sys.argv[2]
    n_particles = sys.argv[3]
    q, p, psi, time = run_integrator(init_type, mode, n_particles)

    output_dir = "integrator"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{mode}_qp_{n_particles}_{machine}.npz")
    np.savez(file_path, q=q, p=p, psi=psi, time=time)