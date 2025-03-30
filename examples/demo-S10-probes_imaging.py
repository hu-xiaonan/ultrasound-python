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
    rf2iq,
    get_fnumber,
    gamma_compress,
)  # noqa: E402

if __name__ == '__main__':
    medium = Medium(rho=1e3, c=1540.0, attenuation=0.5)

    params = {
        'P4-2v': dict(
            description='P4-2v (64-element,\n2.7-MHz phased array)',
            ula=UniformLinearArray(num=64, pitch=300e-6, width=250e-6, height=14e-3),
            props=TransducerProperty(focus=60e-3, baffle=1.75),
            fc=2.72e6,
            bandwidth=0.74,
        ),
        'L11-5v': dict(
            description='L11-5v (128-element,\n7.6-MHz linear array)',
            ula=UniformLinearArray(num=128, pitch=300e-6, width=270e-6, height=5e-3),
            props=TransducerProperty(focus=18e-3, baffle=1.75),
            fc=7.6e6,
            bandwidth=0.77,
        ),
        'L12-3v': dict(
            description='L12-3v (192-element,\n7.5-MHz linear array)',
            ula=UniformLinearArray(num=192, pitch=200e-6, width=170e-6, height=5e-3),
            props=TransducerProperty(focus=20e-3, baffle=1.75),
            fc=7.54e6,
            bandwidth=0.93,
        ),
    }

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

    fig, axs = plt.subplots(1, 3, figsize=(9, 4), layout='constrained')
    for i, probe in enumerate(['P4-2v', 'L11-5v', 'L12-3v']):
        ula = params[probe]['ula']
        props = params[probe]['props']
        fc = params[probe]['fc']
        bandwidth = params[probe]['bandwidth']
        fs = 4*fc
        n_samples = 4096
        tx_delays = get_tx_delays_from_focus(
            ula, medium, [0, -ula.pitch*(ula.num-1)/2/math.tan(math.radians(40))],
        )
        tx_apodization = np.ones(ula.num)
        fnumber = get_fnumber(ula.width, medium.c/(fc*(1+bandwidth/2)))
        time = np.linspace(0, n_samples/fs, n_samples)
        pulse_t = get_tx_pulse_t(n_samples, fs, fc, bandwidth)

        signals_rx_t = get_signals_rx_t(
            ula, props, medium, tx_delays, tx_apodization,
            scatterers, reflectivity, pulse_t, fs,
            db_thresh=-6, workers=None,
        )
        iq_t = rf2iq(signals_rx_t, fc, fs)
        beamformed = beamform(ula, medium, tx_delays, iq_t, fs, points, fnumber, fc)
        echo_img = beamformed.reshape(x_mesh.shape)
        echo_img = np.abs(echo_img)
        gamma_compressed = gamma_compress(echo_img, gamma=0.5)

        axs[i].set_title(params[probe]['description'])
        axs[i].imshow(
            gamma_compressed,
            vmin=0, vmax=1, cmap='gray',
            extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
        )
        axs[i].plot([ula.xmin*100, ula.xmax*100], [0, 0], lw=5, c='C7')
        axs[i].set_ylim(z[-1]*100, 0)
        axs[i].set_xlabel('$x$ (cm)')
        axs[i].set_ylabel('$z$ (cm)')
        for spine in axs[i].spines.values():
            spine.set_visible(False)

    fig.savefig('fig-S10-probes_imaging.png', dpi=300)
