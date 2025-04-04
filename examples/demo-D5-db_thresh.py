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
    time_gain_compensate,
    get_dasmtx,
    beamform_by_dasmtx,
    get_fnumber,
    log_compress,
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
    scatterers = rng.uniform([-4e-2, 0.5e-2], [4e-2, 10e-2], (1000, 2))
    scatterers = scatterers[np.linalg.norm(scatterers-[0, 5e-2], axis=1) > 1e-2]
    reflectivity = np.ones(len(scatterers))

    titles = ['No thresholding', 'Threshold = -40 dB', 'Threshold = -6 dB']
    signals_rx_t = [
        get_signals_rx_t(
            ula, props, medium, tx_delays, tx_apodization,
            scatterers, reflectivity, pulse_t, fs,
            workers=None, show_pbar=True,
        ),
        get_signals_rx_t(
            ula, props, medium, tx_delays, tx_apodization,
            scatterers, reflectivity, pulse_t, fs,
            db_thresh=-40, workers=None, show_pbar=True,
        ),
        get_signals_rx_t(
            ula, props, medium, tx_delays, tx_apodization,
            scatterers, reflectivity, pulse_t, fs,
            db_thresh=-6, workers=None, show_pbar=True,
        ),
    ]
    iq = [rf2iq(rf, fc, fs) for rf in signals_rx_t]
    iq_tgc = [time_gain_compensate(y) for y in iq]

    dasmtx = get_dasmtx(
        ula, medium, tx_delays, n_samples, fs, points, fnumber,
        True, fc,
    )
    beamformed = beamform_by_dasmtx(dasmtx, iq_tgc)
    echo_img = beamformed.reshape((-1, *x_mesh.shape))
    dynamic_range = 40
    echo_img = log_compress(echo_img, dynamic_range=dynamic_range)

    fig, axs = plt.subplots(1, 3, figsize=(9, 4), layout='constrained')
    for i in range(3):
        ax = axs[i]
        ax.set_title(titles[i])
        ax.imshow(
            echo_img[i],
            vmin=-dynamic_range, vmax=0, cmap='gray',
            extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
        )
        ax.set_xlabel('$x$ (cm)')
        ax.set_ylabel('$z$ (cm)')
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.savefig('fig-D5-db_thresh.png', dpi=300)
