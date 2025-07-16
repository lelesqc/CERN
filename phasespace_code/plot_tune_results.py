import numpy as np
import matplotlib.pyplot as plt

import params as par

fft_steps = 32768  # Number of steps for FFT

tune_data = np.load("tune_analysis/fft_results.npz")
action_angle_data = np.load(f"action_angle/tune_a{par.a:.3f}_nu{par.omega_m/par.omega_s:.2f}.npz")

spectrum = tune_data['freqs_list']
spectrum = spectrum[0, :]

freq_mask = (spectrum > 0.0)
spectrum = spectrum[freq_mask]

amplitudes = tune_data['amplitudes']
amplitudes = amplitudes[:, freq_mask]

x = action_angle_data['x']
x = x[:fft_steps]
x_init = x[0, :]

x_mask = (x_init < 12.9)  
x_init = x_init[x_mask]
amplitudes = np.log(amplitudes[x_mask, :])

print(x.shape)
print(np.max(spectrum))

freqs_list= tune_data['freqs_list']
amplit=tune_data['amplitudes']
realpart=np.real(tune_data['spectra'])
imgpart=np.imag(tune_data['spectra'])
plt.figure(figsize=(8, 5))
plt.plot(freqs_list[21, :], realpart[21, :], drawstyle='steps-mid', label='Reale')
plt.plot(freqs_list[21, :], imgpart[21, :], drawstyle='steps-mid', label='Immaginario')
plt.xlabel("Spettro di frequenze")
plt.ylabel("Ampiezza")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


X, Y = np.meshgrid(spectrum, x_init)


plt.figure(figsize=(10, 8))
im = plt.pcolormesh(X, Y, amplitudes, cmap='viridis', shading='nearest',
                    edgecolors=None, linewidth=0.1)
plt.colorbar(im, label='Amplitude')
plt.xlabel('Frequency', fontsize=14)
plt.ylabel('Initial X Position', fontsize=14)
plt.title('Amplitude vs Frequency and Initial Position', fontsize=16)
plt.tight_layout()
plt.show()