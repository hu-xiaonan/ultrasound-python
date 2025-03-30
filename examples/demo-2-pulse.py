import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

os.chdir(Path(__file__).parent.resolve())
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from sonic import get_tx_pulse_t  # noqa: E402

if __name__ == '__main__':
    n_samples = 4096
    fs = 20e6
    duration = n_samples/fs
    time = np.linspace(0, duration, n_samples)
    pulse_t = get_tx_pulse_t(n_samples, fs, fc=2.72e6, bandwidth=0.74)
    freqs = np.fft.fftfreq(n_samples, 1/fs)
    pulse_w = np.fft.fft(pulse_t)
    pulse_w_amp = np.abs(pulse_w)
    dbl = 20*np.log10(pulse_w_amp/pulse_w_amp.max())

    fig, axs = plt.subplots(1, 3, figsize=(9, 2), layout='constrained')

    axs[0].set_title('Transmitted signal')
    axs[0].plot(time*1e6, pulse_t, lw=0.5)
    axs[0].axvspan(0, 2, color='C2', ls='', alpha=0.1)
    axs[0].set_xlabel(r'Time ($\mathrm{\mu s}$)')
    axs[0].set_ylabel('Amplitude (a.u.)')
    axs[0].set_yticks([0])

    axs[1].sharey(axs[0])
    axs[1].set_title('(zoomed)')
    axs[1].plot(time[time <= 2e-6]*1e6, pulse_t[time <= 2e-6], lw=0.5)
    axs[1].axvspan(0, 2, color='C2', ls='', alpha=0.1)
    axs[1].set_xlabel(r'Time ($\mathrm{\mu s}$)')
    axs[1].set_xlim(0, 2)
    axs[1].set_yticks([])

    axs[2].set_title('Magnitude spectrum (dB)')
    axs[2].plot(np.fft.fftshift(freqs)*1e-6, np.fft.fftshift(dbl), lw=0.5)
    axs[2].set_xlabel('Frequency (MHz)')

    fig.savefig('fig-2-pulse.png', dpi=300)
