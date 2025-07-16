import os
import sys
import numpy as np
import params as par
import matplotlib.pyplot as plt
import functions as fn

from scipy.fft import fft, fftfreq
from scipy.signal.windows import hann

def tune_fft(fft_steps):
    data = np.load(f"action_angle/tune_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")
    
    x = data['x']
    y = data['y']

    fft_steps, n_particles = x.shape

    amplitudes = np.zeros((n_particles, fft_steps), dtype=np.float64)
    spectra = np.zeros((n_particles, fft_steps), dtype=np.complex128)
    freqs_list = np.zeros((n_particles, fft_steps), dtype=np.float64)
    interp_tunes = np.zeros(n_particles, dtype=np.float64)

    window = hann(fft_steps)
    
    for i in range(n_particles):
        x_i = x[:, i] * par.lambd
        y_i = y[:, i] / par.lambd

        z_i = (x_i - np.mean(x_i)) - 1j * (y_i - np.mean(y_i))
        z_i_windowed = z_i * window

        spectrum_i = fft(z_i_windowed)
        fft_omega_i = fftfreq(len(z_i), 1 / par.N)

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
            
            co = np.cos(2*np.pi/fft_steps)
            si = np.sin(2*np.pi/fft_steps)
            
            scra1 = co**2 * (p1+p2)**2 - 2*p1*p2*(2*co**2 - co - 1)
            scra2 = (p1 + p2*co)*(p1 - p2)
            scra3 = p1**2 + p2**2 + 2*p1*p2*co
            scra4 = (-scra2 + p2*np.sqrt(scra1)) / scra3
            
            assk = nn + (fft_steps/(2*np.pi)) * np.arcsin(si*scra4)
            delta_f = positive_freqs_i[1] - positive_freqs_i[0]
            freq_interp = positive_freqs_i[0] + assk * delta_f
            interp_tunes[i] = freq_interp

        amplitudes[i, :] = amplitude_i
        spectra[i, :] = spectrum_i
        freqs_list[i, :] = fft_freqs_i
        
    idx = 39

    x = x[::15, :]  # Downsample for better visualization
    y = y[::15, :]

    steps = np.arange(x.shape[0])
    plt.figure(figsize=(8, 6))
    # Linea che unisce i punti
    plt.plot(x[:, idx], y[:, idx], color='gray', linewidth=1, alpha=0.5, label='Trajectory Path')
    # Scatter con punti grandi
    scatter = plt.scatter(x[:, idx], y[:, idx], c=steps, cmap='viridis', s=150, zorder=2)
    # Numeri dentro ogni punto
    for i in range(min(100, x.shape[0])):
        plt.text(x[i, idx], y[i, idx], str(i), color='white', fontsize=10, ha='center', va='center', zorder=3)
    # Evidenzia start/end
    plt.scatter(x[0, idx], y[0, idx], color='red', s=150, label='Start Point', edgecolor='black', zorder=4)
    plt.scatter(x[-1, idx], y[-1, idx], color='blue', s=150, label='End Point', edgecolor='black', zorder=4)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Step index', fontsize=12)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Trajectory of particle with tune = {interp_tunes[idx]:.3f}', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()


    return spectra, freqs_list, interp_tunes, amplitudes

def tune_avg_phase_advance(q, p):
    z = (q - np.mean(q)) - 1j*p
    z_normalized = z / np.abs(z)

    #angles = np.arctan2(p, q - np.mean(q)) 
    angles = np.angle(z_normalized, deg=False)
    angles_unwrapped = np.unwrap(angles, axis=0)
    delta_angles = np.diff(angles_unwrapped, axis=0)
    delta_angles = np.abs(delta_angles) / (2 * np.pi)

    print(delta_angles.shape)

    tunes = np.array([fn.birkhoff_average(delta_angles[:, i]) for i in range(delta_angles.shape[1])])

    #tunes = np.mean(delta_angles, axis=0)
    print(tunes)
    plt.scatter(q[0, :], tunes, color='blue')
    plt.xlabel(r"$\phi$", fontsize=20)
    plt.title(r"Tunes vs $\phi$")
    plt.ylabel("Tune", fontsize=20)
    plt.tight_layout()
    plt.show()





# -------------------------------------

if __name__ == "__main__":
    fft_steps = int(sys.argv[1])

    spectra, freqs_list, tunes_list, amplitudes = fft(fft_steps)

    output_dir = "tune_analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"fft_results.npz")
    np.savez(file_path, spectra=spectra, freqs_list=freqs_list, tunes_list=tunes_list, amplitudes=amplitudes)


