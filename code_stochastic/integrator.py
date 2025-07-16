import os
import sys
import numpy as np

import params as par
import functions as fn

def run_integrator(poincare_mode):
    data = np.load("init_conditions/init_distribution.npz")

    q_init = data['q']
    p_init = data['p']

    q = np.copy(q_init)
    p = np.copy(p_init)

    q_single = None
    p_single = None

    q_sec = np.empty((par.n_steps + 1, *q.shape))
    p_sec = np.empty((par.n_steps + 1, *p.shape))
    sec_count = 0

    step = 0
    psi = par.phi_0
    find_poincare = False
    fixed_params = False

    while not find_poincare:
        q, p = fn.integrator_step(q, p, psi, par.t, par.dt, fn.Delta_q, fn.dV_dq)

        if par.t >= par.T_tot:
            fixed_params = True

        if np.cos(psi) > 1.0 - 1e-3:
            if poincare_mode == "first":
                if q_single is None:
                    q_single = np.copy(q)
                    p_single = np.copy(p)
                    find_poincare = True
            elif poincare_mode == "all":
                q_sec[sec_count] = np.copy(q)
                p_sec[sec_count] = np.copy(p)
                sec_count += 1
                if fixed_params:
                    find_poincare = True
            elif poincare_mode == "last" and fixed_params:
                q_single = np.copy(q)
                p_single = np.copy(p)
                find_poincare = True
                print(f"{np.cos(psi)}, {par.a_lambda(par.t):.5f}, {par.omega_lambda(par.t)/par.omega_s:.5f}")

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
    else:
        q = q_single
        p = p_single

    q = np.array(q)
    p = np.array(p)

    return q, p


# --------------- Save results ----------------


if __name__ == "__main__":
    poincare_mode = sys.argv[1]
    q, p = run_integrator(poincare_mode)

    output_dir = "integrator"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"evolved_qp_{poincare_mode}.npz")
    np.savez(file_path, q=q, p=p)