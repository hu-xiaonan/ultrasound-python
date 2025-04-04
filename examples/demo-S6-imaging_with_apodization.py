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
    beamform_by_dasmtx,
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

    tx_apodization_rect = np.ones(ula.num)
    tx_apodization_hanning = np.hanning(ula.num)
    tx_apodization_subaperture = np.asarray([1.0]*16+[0.0]*(64-16))

    titles = [
        'No apodization',
        'Hanning window',
        'Subaperture',
    ]
    signals_rx_t = [
        get_signals_rx_t(
            ula, props, medium, tx_delays, tx_apodization_rect,
            scatterers, reflectivity, pulse_t, fs,
            workers=-1, show_pbar=True,
        ),
        get_signals_rx_t(
            ula, props, medium, tx_delays, tx_apodization_hanning,
            scatterers, reflectivity, pulse_t, fs,
            workers=-1, show_pbar=True,
        ),
        get_signals_rx_t(
            ula, props, medium, tx_delays, tx_apodization_subaperture,
            scatterers, reflectivity, pulse_t, fs,
            workers=-1, show_pbar=True,
        ),
    ]
    iq = [rf2iq(rf, fc, fs) for rf in signals_rx_t]

    dasmtx = get_dasmtx(
        ula, medium, tx_delays, n_samples, fs, points, fnumber,
        True, fc,
    )
    beamformed = beamform_by_dasmtx(dasmtx, iq)
    echo_img = beamformed.reshape((-1, *x_mesh.shape))
    echo_img = gamma_compress(echo_img, gamma=0.5)

    fig, axs = plt.subplots(1, 3, figsize=(9, 4), layout='constrained')
    for i in range(3):
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
    fig.savefig('fig-S6-imaging_with_apodization.png', dpi=300)
