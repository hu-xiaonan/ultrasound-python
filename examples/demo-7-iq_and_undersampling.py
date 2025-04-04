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
    beamform,
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
    signals_rx_t = get_signals_rx_t(
        ula, props, medium, tx_delays, tx_apodization,
        scatterers, reflectivity, pulse_t, fs,
        workers=-1, show_pbar=True,
    )

    undersampling_factor = 5
    fs_us = fs/undersampling_factor
    signals_rx_t_us = signals_rx_t[:, ::undersampling_factor]

    iq_t = rf2iq(signals_rx_t, fc, fs)
    iq_t_us = rf2iq(signals_rx_t_us, fc, fs_us)

    titles = [
        'Using RF',
        'Using I/Q demodulation of RF',
        'Using 5x-undersampled RF',
        'Using I/Q demodulation\nof 5x-undersampled RF',
    ]
    beamformed = [
        beamform(ula, medium, tx_delays, signals_rx_t, fs, points, fnumber),
        beamform(ula, medium, tx_delays, iq_t, fs, points, fnumber, fc),
        beamform(ula, medium, tx_delays, signals_rx_t_us, fs_us, points, fnumber),
        beamform(ula, medium, tx_delays, iq_t_us, fs_us, points, fnumber, fc),
    ]

    echo_img = np.reshape(beamformed, (-1, *x_mesh.shape))
    echo_img = gamma_compress(echo_img, gamma=0.5)

    fig, axs = plt.subplots(1, 4, figsize=(12, 4), layout='constrained')
    for i in range(4):
        ax = axs[i]
        ax.set_title(titles[i])
        ax.imshow(
            echo_img[i],
            vmin=0, vmax=1, cmap='gray',
            extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
        )
        ax.set_xlabel('$x$ (cm)')
        ax.set_ylabel('$z$ (cm)')
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.savefig('fig-7-iq_and_undersampling.png', dpi=300)
