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
    get_dasmtx,
    beamform,
    get_fnumber,
    rf2iq,
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

    signals_rx_t = get_signals_rx_t(
        ula, props, medium, tx_delays, tx_apodization,
        scatterers, reflectivity, pulse_t, fs,
        workers=-1, show_pbar=True,
    )
    iq_t = rf2iq(signals_rx_t, fc, fs)

    dasmtx = get_dasmtx(
        ula, medium, tx_delays, n_samples, fs, points, fnumber,
        True, fc,
    )
    beamformed_optim_aperture = beamform(ula, medium, tx_delays, iq_t, fs, points, fnumber, fc)
    beamformed_full_aperture = beamform(ula, medium, tx_delays, iq_t, fs, points, fnumber=None, fc=fc)

    echo_img_optim_aperture = beamformed_optim_aperture.reshape(x_mesh.shape)
    echo_img_full_aperture = beamformed_full_aperture.reshape(x_mesh.shape)
    echo_img_optim_aperture = gamma_compress(echo_img_optim_aperture, gamma=0.5)
    echo_img_full_aperture = gamma_compress(echo_img_full_aperture, gamma=0.5)

    fig, axs = plt.subplots(1, 2, figsize=(6, 4), layout='constrained')
    axs[0].set_title('Beamforming\nwith optimized aperture')
    axs[0].imshow(
        echo_img_optim_aperture,
        vmin=0, vmax=1, cmap='gray',
        extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
    )
    axs[1].set_title('Beamforming\nwith full aperture')
    axs[1].imshow(
        echo_img_full_aperture,
        vmin=0, vmax=1, cmap='gray',
        extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
    )
    for ax in axs:
        axs[1].set_xlabel('$x$ (cm)')
        axs[1].set_ylabel('$z$ (cm)')
        for spine in axs[1].spines.values():
            spine.set_visible(False)
    fig.savefig('fig-S7-fnumber.png', dpi=300)
