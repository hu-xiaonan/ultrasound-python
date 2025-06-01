# Copyright (c) 2023-2025 Hu Xiaonan
# License: MIT License

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
    beamform,
    get_fnumber,
    time_gain_compensate,
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

    rng = np.random.default_rng(202307)
    scatterers = rng.uniform([-4e-2, 0.5e-2], [4e-2, 10e-2], (100, 2))
    reflectivity = np.ones(len(scatterers))

    signals_rx_t = get_signals_rx_t(
        ula, props, medium, tx_delays, tx_apodization,
        scatterers, reflectivity, pulse_t, fs,
        workers=-1, show_pbar=True,
    )
    signals_tgc = time_gain_compensate(signals_rx_t)

    beamformed = beamform(ula, medium, tx_delays, signals_rx_t, fs, points, fnumber)
    echo_img = beamformed.reshape(x_mesh.shape)
    echo_img = np.abs(echo_img)

    beamformed_tgc = beamform(ula, medium, tx_delays, signals_tgc, fs, points, fnumber)
    echo_img_tgc = beamformed_tgc.reshape(x_mesh.shape)
    echo_img_tgc = np.abs(echo_img_tgc)

    gamma_compressed = gamma_compress(echo_img, gamma=0.5)
    gamma_compressed_tgc = gamma_compress(echo_img_tgc, gamma=0.5)

    fig, axs = plt.subplots(1, 4, figsize=(12, 4), width_ratios=[2, 2, 3, 3], layout='constrained')

    signals_rx_t_normalized = signals_rx_t/np.ptp(signals_rx_t)
    signals_tgc_normalized = signals_tgc/np.ptp(signals_tgc)

    axs[0].set_title('Received signals (a.u.)')
    for i in range(ula.num):
        axs[0].plot(signals_rx_t_normalized[i]+i+1, time*1e6, c='k', lw=0.5)
    axs[0].invert_yaxis()
    axs[0].set_xlabel('Element #')
    axs[0].set_xticks([1, 16, 32, 48, 64])
    axs[0].set_ylabel(r'Time ($\mathrm{\mu s}$)')

    axs[1].set_title('TGC signals (a.u.)')
    for i in range(ula.num):
        axs[1].plot(signals_tgc_normalized[i]+i+1, time*1e6, c='k', lw=0.5)
    axs[1].invert_yaxis()
    axs[1].set_xlabel('Element #')
    axs[1].set_xticks([1, 16, 32, 48, 64])
    axs[1].set_ylabel(r'Time ($\mathrm{\mu s}$)')

    axs[2].set_title('Beamformed image\n(gamma compressed)')
    axs[2].imshow(
        gamma_compressed,
        vmin=0, vmax=1, cmap='gray',
        extent=(x.min()*100, x.max()*100, z.min()*100, z.max()*100),
    )
    axs[2].set_xlabel('$x$ (cm)')
    axs[2].set_ylabel('$z$ (cm)')
    for spine in axs[2].spines.values():
        spine.set_visible(False)

    axs[3].set_title('Beamformed image of TGC\n(gamma compressed)')
    axs[3].imshow(
        gamma_compressed_tgc,
        vmin=0, vmax=1, cmap='gray',
        extent=(x.min()*100, x.max()*100, z.min()*100, z.max()*100),
    )
    axs[3].set_xlabel('$x$ (cm)')
    axs[3].set_ylabel('$z$ (cm)')
    for spine in axs[3].spines.values():
        spine.set_visible(False)

    fig.savefig('fig-8-tgc.png', dpi=300)
