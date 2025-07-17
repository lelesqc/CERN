import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import params as par
import functions as fn

def run_integrator(mode, fft_steps):
    par.fft_steps = fft_steps

    data = np.load("init_conditions/qp.npz")

    q_init = data['q']
    p_init = data['p']

    q = np.copy(q_init)
    p = np.copy(p_init)

    n_particles = len(q)

    psi = par.phi_0

    initial_angles = np.array([np.arctan2(p[i], q[i]) for i in range(n_particles)])
    cumulative_angles = np.zeros(n_particles)
    loop_completed = np.zeros(n_particles, dtype=bool)
    first_loops = [[] for _ in range(n_particles)]

    if mode == "tune":
        q_traj = np.zeros((fft_steps, len(q)), dtype=np.float32)
        p_traj = np.zeros((fft_steps, len(p)), dtype=np.float32)
        step_count = 0        

        while step_count < fft_steps:
            q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)
            q_traj[step_count] = q
            p_traj[step_count] = p            
            step_count += 1

            #if np.cos(psi) > 1.0 - 1e-3:
            #    q_traj[step_count] = q
            #    p_traj[step_count] = p
            #    step_count += 1

            psi += par.omega_m * par.dt
            par.t += par.dt

        q = q_traj
        p = p_traj

    elif mode == "phasespace":
        areas = np.zeros(n_particles)
        trajs = np.zeros((par.n_steps, n_particles, 2), dtype=np.float32)

        for step in range(par.n_steps):
            q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)
            trajs[step, :, 0] = q
            trajs[step, :, 1] = p
            psi += par.omega_m * par.dt
            par.t += par.dt

        idx = 5
        traj = trajs[:, idx, :]
        traj = fn.order_points(traj)
        area = fn.shoelace(traj)
        indices = np.arange(len(traj))
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(traj[:, 0], traj[:, 1], c=indices, cmap='viridis', s=30, label='Step', alpha=0.95)
        plt.scatter(traj[0, 0], traj[0, 1], color='lime', s=60, label='Start', edgecolor='k', zorder=3)
        plt.scatter(traj[-1, 0], traj[-1, 1], color='red', s=60, label='End', edgecolor='k', zorder=3)
        plt.colorbar(scatter, label='Ordine punto nella traiettoria')
        plt.title(f"Area = {areas[idx]:.6f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

# --------------- Save results ----------------

if __name__ == "__main__":
    mode = sys.argv[1]
    fft_steps = int(sys.argv[2])
    run_integrator(mode, fft_steps)

    output_dir = "integrator"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"evolved_qp_{mode}.npz")