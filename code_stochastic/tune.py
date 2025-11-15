import os
import numpy as np
from scipy.fft import fft, fftfreq

import params as par

def tune_calculation():
    a_start = par.a_lambda(par.T_percent)
    omega_start = par.omega_lambda(par.T_percent)
    a_end = par.a_lambda(par.T_tot)
    omega_end = par.omega_lambda(par.T_tot)
    
    a_start_str = f"{a_start:.3f}"
    omega_start_str = f"{omega_start:.2f}"
    a_end_str = f"{a_end:.3f}"
    omega_end_str = f"{omega_end:.2f}"

    str_title = f"a{a_start_str}-{a_end_str}_nu{float(omega_start_str)/par.omega_s:.2f}-{float(omega_end_str)/par.omega_s:.2f}"

    data = np.load(f"action_angle/tune_{str_title}.npz")

    x = data['x']
    y = data['y']

    n_particles = int(len(x) / par.n_steps) 

    spectra = []
    freqs_list = []
    tunes_list = []

    n_extra = 4096
    for i in range(n_particles):
        x_i = x[i * par.n_steps : (i + 1) * par.n_steps]
        y_i = y[i * par.n_steps : (i + 1) * par.n_steps]

        z_i = x_i - 1j * y_i

        n = np.arange(par.n_steps)
        chi = 2 * np.sin(np.pi * n / n_extra)**2

        z_i_windowed = z_i * chi

        spectrum_i = fft(z_i_windowed)
        fft_omega_i = fftfreq(par.n_steps, par.dt)
        fft_freqs_i = fft_omega_i / par.omega_s

        abs_spec = np.abs(spectrum_i)
        freqs_pos = fft_freqs_i
        idx_max = np.argmax(abs_spec)
        tune_i = freqs_pos[idx_max]

        spectra.append(spectrum_i)
        freqs_list.append(fft_freqs_i)
        tunes_list.append(tune_i)

    spectra = np.array(spectra)
    freqs_list = np.array(freqs_list)
    tunes_list = np.array(tunes_list)          

    return spectra, freqs_list, tunes_list


# -------------------------------------


if __name__ == "__main__":
    spectra, freqs_list, tunes_list = tune_calculation()

    output_dir = "tune_analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, "fft_results.npz")
    np.savez(file_path, spectra=spectra, freqs_list=freqs_list, tunes_list=tunes_list)
