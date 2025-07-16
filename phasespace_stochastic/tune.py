import os
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal.windows import hann
import params as par
import matplotlib.pyplot as plt

def tune_calculation():
    fft_steps = 4096

    data = np.load(f"action_angle/tune_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")
    
    x = data['x']
    y = data['y']

    x = x[:fft_steps]
    y = y[:fft_steps]

    n_steps, n_particles = x.shape

    amplitudes = np.zeros((n_particles, n_steps), dtype=np.float64)
    spectra = np.zeros((n_particles, n_steps), dtype=np.complex128)
    freqs_list = np.zeros((n_particles, n_steps), dtype=np.float64)
    raw_tunes = np.zeros(n_particles, dtype=np.float64)
    interp_tunes = np.zeros(n_particles, dtype=np.float64)

    window = hann(n_steps)
    
    for i in range(n_particles):
        x_i = x[:, i] * par.lambd
        y_i = y[:, i] / par.lambd

        z_i = (x_i - np.mean(x_i)) - 1j * (y_i - np.mean(y_i))
        z_i_windowed = z_i * window

        spectrum_i = fft(z_i_windowed)
        #fft_omega_i = fftfreq(len(z_i), par.dt * par.omega_s)
        fft_omega_i = fftfreq(len(z_i), 1 / par.N)

        #fft_freqs_i = fft_omega_i / par.omega_s * 2 * np.pi
        fft_freqs_i = fft_omega_i

        amplitude_i = np.abs(spectrum_i)
        positive_freq_mask = fft_freqs_i > 0
        positive_freqs_i = fft_freqs_i[positive_freq_mask]
        positive_ampls = amplitude_i[positive_freq_mask]

        idx_max = np.argmax(positive_ampls)
        tune_i = positive_freqs_i[idx_max]
        
        # interpolation
        if idx_max > 0 and idx_max < len(positive_ampls)-1:
            cf1 = positive_ampls[idx_max-1]
            cf2 = positive_ampls[idx_max]
            cf3 = positive_ampls[idx_max+1]
            
            if cf3 > cf1:
                p1, p2 = cf2, cf3
                nn = idx_max
            else:
                p1, p2 = cf1, cf2
                nn = idx_max - 1
            
            co = np.cos(2*np.pi/n_steps)
            si = np.sin(2*np.pi/n_steps)
            
            scra1 = co**2 * (p1+p2)**2 - 2*p1*p2*(2*co**2 - co - 1)
            scra2 = (p1 + p2*co)*(p1 - p2)
            scra3 = p1**2 + p2**2 + 2*p1*p2*co
            scra4 = (-scra2 + p2*np.sqrt(scra1)) / scra3
            
            assk = nn + (n_steps/(2*np.pi)) * np.arcsin(si*scra4)
            delta_f = positive_freqs_i[1] - positive_freqs_i[0]
            freq_interp = positive_freqs_i[0] + assk * delta_f
            interp_tunes[i] = freq_interp

        amplitudes[i, :] = amplitude_i
        spectra[i, :] = spectrum_i
        freqs_list[i, :] = fft_freqs_i

    plt.figure(figsize=(8, 5))
    plt.plot(freqs_list[27, :], amplitudes[27, :], drawstyle='steps-mid')
    plt.xlabel("Spettro di frequenze")
    plt.ylabel("Ampiezza")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(len(spectra[27, :]))

    return spectra, freqs_list, interp_tunes, amplitudes

# -------------------------------------

if __name__ == "__main__":
    spectra, freqs_list, tunes_list, amplitudes = tune_calculation()

    output_dir = "tune_analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, "fft_results.npz")
    np.savez(file_path, spectra=spectra, freqs_list=freqs_list, tunes_list=tunes_list, amplitudes=amplitudes)
