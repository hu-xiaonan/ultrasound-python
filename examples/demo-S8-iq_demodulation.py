import math
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

os.chdir(Path(__file__).parent.resolve())
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from sonic import (
    UniformLinearArray,
    TransducerProperty,
    Medium,
    get_tx_delays_from_focus,
    get_tx_pulse_t,
    get_signals_rx_t,
    rf2iq,
)  # noqa: E402

if __name__ == '__main__':
    ula = UniformLinearArray(num=64, pitch=300e-6, width=250e-6, height=14e-3)
    props = TransducerProperty(focus=60e-3, baffle=1.75)
    medium = Medium(rho=1e3, c=1540.0, attenuation=0.5)
    tx_delays = get_tx_delays_from_focus(
        ula, medium, [0, -ula.pitch*(ula.num-1)/2/math.tan(math.radians(40))],
    )
    tx_apodization = np.ones(ula.num)

    fc = 2.72e6
    bandwidth = 0.74
    n_samples = 2048
    fs = 10e6
    time = np.linspace(0, n_samples/fs, n_samples)
    pulse_t = get_tx_pulse_t(n_samples, fs, fc, bandwidth)
    freqs = np.fft.fftfreq(n_samples, 1/fs)
    spectrum_tx = np.fft.fft(pulse_t)

    rng = np.random.default_rng(202307)
    scatterers = rng.uniform((-4e-2, 0.1e-2), (4e-2, 10e-2), (100, 2))
    reflectivity = np.ones(len(scatterers))

    # RF signals.
    signals_rx_t = get_signals_rx_t(
        ula, props, medium, tx_delays, tx_apodization,
        scatterers, reflectivity, pulse_t, fs,
        workers=-1, show_pbar=True,
    )
    spectrum_rx = np.fft.fft(signals_rx_t, axis=1)

    # I/Q signals.
    iq_t = rf2iq(signals_rx_t, fc, fs)
    spectrum_iq = np.fft.fft(iq_t, axis=1)

    # 5x-undersampled RF signals.
    undersampling_factor = 5
    fs_us = fs/undersampling_factor
    time_us = time[::undersampling_factor]
    n_samples_us = len(time_us)
    freqs_us = np.fft.fftfreq(n_samples_us, 1/fs_us)
    signals_rx_t_us = signals_rx_t[:, ::undersampling_factor]
    spectrum_rx_us = np.fft.fft(signals_rx_t_us, axis=1)

    # I/Q of 5x-undersampled RF signals.
    iq_t_us = rf2iq(signals_rx_t_us, fc, fs_us)
    spectrum_iq_us = np.fft.fft(iq_t_us, axis=1)

    freqs = np.fft.fftshift(freqs)
    freqs_us = np.fft.fftshift(freqs_us)
    spectrum_tx = np.fft.fftshift(spectrum_tx)
    spectrum_rx = np.fft.fftshift(spectrum_rx, axes=1)
    spectrum_iq = np.fft.fftshift(spectrum_iq, axes=1)
    spectrum_rx_us = np.fft.fftshift(spectrum_rx_us, axes=1)
    spectrum_iq_us = np.fft.fftshift(spectrum_iq_us, axes=1)

    spectrum_tx_mag = np.abs(spectrum_tx)
    spectrum_rx_mag = np.abs(spectrum_rx)
    spectrum_iq_mag = np.abs(spectrum_iq)
    spectrum_rx_us_mag = np.abs(spectrum_rx_us)
    spectrum_iq_us_mag = np.abs(spectrum_iq_us)

    dbl_tx = 20*np.log10(spectrum_tx_mag/spectrum_tx_mag.max())
    dbl_rx = 20*np.log10(spectrum_rx_mag/spectrum_rx_mag.max())
    dbl_iq = 20*np.log10(spectrum_iq_mag/spectrum_iq_mag.max())
    dbl_rx_us = 20*np.log10(spectrum_rx_us_mag/spectrum_rx_us_mag.max())
    dbl_iq_us = 20*np.log10(spectrum_iq_us_mag/spectrum_iq_us_mag.max())

    fig, axs = plt.subplots(5, 3, figsize=(12, 7), sharex='col', layout='constrained')

    ELEM_FIVE = 4
    zoom_t0, zoom_t1 = 0e-6, 25e-6
    interval = (time >= zoom_t0) & (time <= zoom_t1)

    axs[0, 0].set_title('Transmitted')
    axs[0, 0].plot(time*1e6, pulse_t, lw=0.5)
    axs[1, 0].set_title('RF (elem#5)')
    axs[1, 0].plot(time*1e6, signals_rx_t[ELEM_FIVE], lw=0.5)
    axs[2, 0].set_title('I/Q (elem#5)')
    axs[2, 0].plot(time*1e6, iq_t[ELEM_FIVE].real, label='Real', lw=0.5)
    axs[2, 0].plot(time*1e6, iq_t[ELEM_FIVE].imag, label='Imag', lw=0.5)
    axs[2, 0].legend(loc='upper right', ncol=2)
    axs[3, 0].set_title('5x-undersampled RF (elem#5)')
    axs[3, 0].plot(time_us*1e6, signals_rx_t_us[ELEM_FIVE], lw=0.5)
    axs[4, 0].set_title('I/Q of 5x-undersampled RF (elem#5)')
    axs[4, 0].plot(time_us*1e6, iq_t_us[ELEM_FIVE].real, label='Real', lw=0.5)
    axs[4, 0].plot(time_us*1e6, iq_t_us[ELEM_FIVE].imag, label='Imag', lw=0.5)
    axs[4, 0].legend(loc='upper right', ncol=2)
    axs[4, 0].set_xlabel(r'Time ($\mathrm{\mu s}$)')
    for ax in axs[:, 0]:
        ax.axvspan(zoom_t0*1e6, zoom_t1*1e6, color='C2', ls='', alpha=0.1)
        ax.set_yticks([])
        ax.set_ylabel('Amp. (a.u.)')

    axs[0, 1].plot(time[interval]*1e6, pulse_t[interval], lw=0.5)
    axs[1, 1].plot(time[interval]*1e6, signals_rx_t[ELEM_FIVE, interval], lw=0.5)
    axs[2, 1].plot(time[interval]*1e6, iq_t[ELEM_FIVE, interval].real, lw=0.5)
    axs[2, 1].plot(time[interval]*1e6, iq_t[ELEM_FIVE, interval].imag, lw=0.5)
    axs[3, 1].plot(time_us*1e6, signals_rx_t_us[ELEM_FIVE], lw=0.5)
    axs[4, 1].plot(time_us*1e6, iq_t_us[ELEM_FIVE].real, lw=0.5)
    axs[4, 1].plot(time_us*1e6, iq_t_us[ELEM_FIVE].imag, lw=0.5)
    axs[4, 1].set_xlabel(r'Time ($\mathrm{\mu s}$)')
    for ax, ax_zoom in axs[:, 0:2]:
        ax_zoom.set_title('(zoomed)')
        ax_zoom.axvspan(zoom_t0*1e6, zoom_t1*1e6, color='C2', ls='', alpha=0.1)
        ax_zoom.set_xlim(zoom_t0*1e6, zoom_t1*1e6)
        ax_zoom.sharey(ax)

    axs[0, 2].plot(freqs*1e-6, dbl_tx, lw=0.5)
    axs[1, 2].plot(freqs*1e-6, dbl_rx[4], lw=0.5)
    axs[2, 2].plot(freqs*1e-6, dbl_iq[4], lw=0.5)
    axs[3, 2].plot(freqs_us*1e-6, dbl_rx_us[4], lw=0.5)
    axs[4, 2].plot(freqs_us*1e-6, dbl_iq_us[4], lw=0.5)
    for ax in axs[:, 2]:
        ax.set_title('Magnitude spectrum (dB)')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_yticks([-200, -100, 0])

    fig.savefig('fig-S8-iq_demodulation.png', dpi=300)
