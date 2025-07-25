import os
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import params as par
import functions as fn
import action_angle as aa

# ------ Inizializzazione -------
starting_data = np.load("init_conditions/init_distribution.npz")
q_init = starting_data['q']
p_init = starting_data['p']

print(q_init.shape, p_init.shape)

n_particles = len(q_init)

q_traj = np.zeros((par.n_steps, n_particles), dtype=np.float16)
p_traj = np.zeros((par.n_steps, n_particles), dtype=np.float16)

q_traj[0, :] = q_init
p_traj[0, :] = p_init

q = np.copy(q_traj[0, :])
p = np.copy(p_traj[0, :])

psi = 0
angles = np.zeros((par.n_steps // 7, n_particles), dtype=np.float16)
angles[0, :] = np.arctan2(p, q - np.pi)

# ------ Integratore -------

step = 0
while par.t < par.T_percent: 
    q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)
    angles[step + 1, :] = np.arctan2(p, q - np.pi)
    q_traj[step + 1, :] = q
    p_traj[step + 1, :] = p
    psi += par.omega_lambda(par.t) * par.dt
    par.t += par.dt

    step += 1

angles = angles[:step, :]
angles = np.unwrap(angles, axis=0)

# ------ Estrazione dei loop chiusi -------
q_loops = []
p_loops = []

for i in tqdm(range(n_particles)):
    angle_i = angles[:, i]
    delta_angle = angle_i - angle_i[0]
    
    loop_start = 0
    particle_q_loops = []
    particle_p_loops = []

    while True:
        mask = np.abs(delta_angle[loop_start:] - delta_angle[loop_start]) >= 2 * np.pi
        if not np.any(mask):
            break
        idx = np.argmax(mask) + loop_start

        q_segment = q_traj[loop_start:idx, i]
        p_segment = p_traj[loop_start:idx, i]

        q_closed = np.append(q_segment, q_segment[0])
        p_closed = np.append(p_segment, p_segment[0])

        particle_q_loops.append(q_closed)
        particle_p_loops.append(p_closed)

        loop_start = idx

    q_loops.append(particle_q_loops)
    p_loops.append(particle_p_loops)

# ------ Funzione da parallelizzare -------
def compute_xy_and_action(q_loop, p_loop):
    n_steps = len(q_loop)
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    
    for i in range(n_steps):
        h_0 = fn.H0_for_action_angle(q_loop[i], p_loop[i])
        kappa_squared = 0.5 * (1 + h_0 / (par.A**2))
        if 0 < kappa_squared < 1:
            Q = (q_loop[i] + np.pi) / par.lambd
            P = par.lambd * p_loop[i]
            action, theta = fn.compute_action_angle(kappa_squared, P)
            x[i] = np.sqrt(2 * action) * np.cos(theta)
            y[i] = - np.sqrt(2 * action) * np.sin(theta) * np.sign(q_loop[i] - np.pi)

    # calcolo area e action iniziale
    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])
    area = 0.5 * np.abs(np.dot(x_closed[:-1], y_closed[1:]) - np.dot(y_closed[:-1], x_closed[1:]))
    action_final = area / (2 * np.pi)

    return x, y, action_final

# ------ Parallelizzazione -------
input_list = [
    (np.array(q_loop), np.array(p_loop))
    for particle_q_loops, particle_p_loops in zip(q_loops, p_loops)
    for q_loop, p_loop in zip(particle_q_loops, particle_p_loops)
]

particle_indices = []

for particle_idx, (particle_q_loops, particle_p_loops) in enumerate(zip(q_loops, p_loops)):
    for q_loop, p_loop in zip(particle_q_loops, particle_p_loops):
        input_list.append((np.array(q_loop), np.array(p_loop)))
        particle_indices.append(particle_idx)

particle_indices = np.array(particle_indices)

with tqdm_joblib(tqdm(desc="Calcolo azioni", total=len(input_list))) as progress_bar:
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(compute_xy_and_action)(q_loop, p_loop)
        for q_loop, p_loop in input_list
    )

# ------ Ricostruzione risultati -------
x_list = [res[0] for res in results]
y_list = [res[1] for res in results]
init_actions = np.array([res[2] for res in results])

os.makedirs("actions_stuff")
np.savez("./actions_stuff/actions_first_part.npz", init_actions=init_actions, particle_indices=particle_indices)

print(f"Initial actions: {init_actions}")

