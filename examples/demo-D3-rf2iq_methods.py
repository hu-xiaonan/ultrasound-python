import math
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

os.chdir(Path(__file__).parent.resolve())
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from sonic import (
    UniformLinearArray,
    TransducerProperty,
    Medium,
    get_tx_delays_from_focus,
    get_tx_pulse_t,
    get_signals_rx_t,
    beamform,
    rf2iq,
    get_fnumber,
    gamma_compress,
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
    fnumber = get_fnumber(ula.width, medium.c/(fc*(1+bandwidth/2)))
    n_samples = 2048
    fs = 10e6
    time = np.linspace(0, n_samples/fs, n_samples)
    pulse_t = get_tx_pulse_t(n_samples, fs, fc, bandwidth)

    x = np.linspace(-4e-2, 4e-2, 400+1)
    z = np.linspace(0.1e-2, 10e-2, 500+1)
    x_mesh, z_mesh = np.meshgrid(x, z, indexing='xy')
    points = np.transpose([x_mesh.reshape((-1,)), z_mesh.reshape((-1,))])

    scatterers = np.asarray(
        [
            [x, z]
            for x in np.linspace(-3e-2, 3e-2, 13)
            for z in np.linspace(1e-2, 9e-2, 17)
        ]
    )
    reflectivity = np.ones(len(scatterers))

    # RF signals.
    signals_rx_t = get_signals_rx_t(
        ula, props, medium, tx_delays, tx_apodization,
        scatterers, reflectivity, pulse_t, fs,
        workers=-1, show_pbar=True,
    )

    # 5x-undersampled RF signals.
    undersampling_factor = 5
    fs_us = fs/undersampling_factor
    time_us = time[::undersampling_factor]
    n_samples_us = len(time_us)
    signals_rx_t_us = signals_rx_t[:, ::undersampling_factor]

    # I/Q signals.
    titles = [
        'Downmixing + Low-pass\n(fine-sampled)',
        'Hilbert + Downmixing\n(fine-sampled)',
        'Downmixing + Low-pass\n(undersampled)',
        'Hilbert + Downmixing\n(undersampled)',
    ]
    iq_t = rf2iq(signals_rx_t, fc, fs)
    iq_t_hilb = hilbert(signals_rx_t, axis=1)*np.exp(-1j*math.tau*fc*time)
    iq_t_us = rf2iq(signals_rx_t_us, fc, fs_us)
    iq_t_us_hilb = hilbert(signals_rx_t_us, axis=1)*np.exp(-1j*math.tau*fc*time_us)

    freqs = np.fft.fftfreq(n_samples, 1/fs)
    spectrum_iq = np.fft.fft(iq_t, axis=1)
    spectrum_iq_hilb = np.fft.fft(iq_t_hilb, axis=1)
    freqs_us = np.fft.fftfreq(n_samples_us, 1/fs_us)
    spectrum_iq_us = np.fft.fft(iq_t_us, axis=1)
    spectrum_iq_us_hilb = np.fft.fft(iq_t_us_hilb, axis=1)
    
    freqs = np.fft.fftshift(freqs)
    freqs_us = np.fft.fftshift(freqs_us)
    spectrum_iq = np.fft.fftshift(spectrum_iq, axes=1)
    spectrum_iq_hilb = np.fft.fftshift(spectrum_iq_hilb, axes=1)
    spectrum_iq_us = np.fft.fftshift(spectrum_iq_us, axes=1)
    spectrum_iq_us_hilb = np.fft.fftshift(spectrum_iq_us_hilb, axes=1)

    spectrum_iq_mag = np.abs(spectrum_iq)
    spectrum_iq_hilb_mag = np.abs(spectrum_iq_hilb)
    spectrum_iq_us_mag = np.abs(spectrum_iq_us)
    spectrum_iq_us_hilb_mag = np.abs(spectrum_iq_us_hilb)

    dbl_iq = 20*np.log10(spectrum_iq_mag/spectrum_iq_mag.max())
    dbl_iq_hilb = 20*np.log10(spectrum_iq_hilb_mag/spectrum_iq_hilb_mag.max())
    dbl_iq_us = 20*np.log10(spectrum_iq_us_mag/spectrum_iq_us_mag.max())
    dbl_iq_us_hilb = 20*np.log10(spectrum_iq_us_hilb_mag/spectrum_iq_us_hilb_mag.max())

    beamformed_iq = beamform(ula, medium, tx_delays, iq_t, fs, points, fnumber, fc)
    beamformed_iq_hilb = beamform(ula, medium, tx_delays, iq_t_hilb, fs, points, fnumber, fc)
    beamformed_iq_us = beamform(ula, medium, tx_delays, iq_t_us, fs_us, points, fnumber, fc)
    beamformed_iq_us_hilb = beamform(ula, medium, tx_delays, iq_t_us_hilb, fs_us, points, fnumber, fc)

    echo_img_iq = np.reshape(beamformed_iq, x_mesh.shape)
    echo_img_iq_hilb = np.reshape(beamformed_iq_hilb, x_mesh.shape)
    echo_img_iq_us = np.reshape(beamformed_iq_us, x_mesh.shape)
    echo_img_iq_us_hilb = np.reshape(beamformed_iq_us_hilb, x_mesh.shape)

    echo_img_iq = gamma_compress(np.abs(echo_img_iq), gamma=0.5)
    echo_img_iq_hilb = gamma_compress(np.abs(echo_img_iq_hilb), gamma=0.5)
    echo_img_iq_us = gamma_compress(np.abs(echo_img_iq_us), gamma=0.5)
    echo_img_iq_us_hilb = gamma_compress(np.abs(echo_img_iq_us_hilb), gamma=0.5)

    ELEM_FIVE = 5
    fig, axs = plt.subplots(2, 4, figsize=(12, 5), height_ratios=[1, 4], sharey='row', layout='constrained')

    axs[0, 0].set_title(titles[0])
    axs[0, 0].plot(freqs/1e6, dbl_iq[ELEM_FIVE], lw=0.5)
    axs[0, 0].set_xlabel('Frequency (MHz)')
    axs[0, 0].set_ylabel('dB')
    axs[1, 0].imshow(
        echo_img_iq, vmin=0, vmax=1, cmap='gray',
        extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
    )
    axs[1, 0].set_xlabel('$x$ (cm)')
    axs[1, 0].set_ylabel('$z$ (cm)')

    axs[0, 1].set_title(titles[1])
    axs[0, 1].plot(freqs/1e6, dbl_iq_hilb[ELEM_FIVE], lw=0.5)
    axs[0, 1].set_xlabel('Frequency (MHz)')
    axs[0, 1].sharex(axs[0, 0])
    axs[1, 1].imshow(
        echo_img_iq_hilb, vmin=0, vmax=1, cmap='gray',
        extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
    )
    axs[1, 1].set_xlabel('$x$ (cm)')

    axs[0, 2].set_title(titles[2])
    axs[0, 2].plot(freqs_us/1e6, dbl_iq_us[ELEM_FIVE], lw=0.5)
    axs[0, 2].set_xlabel('Frequency (MHz)')
    axs[0, 2].sharex(axs[0, 0])
    axs[1, 2].imshow(
        echo_img_iq_us, vmin=0, vmax=1, cmap='gray',
        extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
    )
    axs[1, 2].set_xlabel('$x$ (cm)')

    axs[0, 3].set_title(titles[3])
    axs[0, 3].plot(freqs_us/1e6, dbl_iq_us_hilb[ELEM_FIVE], lw=0.5)
    axs[0, 3].set_xlabel('Frequency (MHz)')
    axs[0, 3].sharex(axs[0, 0])
    axs[1, 3].imshow(
        echo_img_iq_us_hilb, vmin=0, vmax=1, cmap='gray',
        extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
    )
    axs[1, 3].set_xlabel('$x$ (cm)')

    for ax in axs[1, :]:
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.savefig('fig-D3-rf2iq_methods.png', dpi=300)
