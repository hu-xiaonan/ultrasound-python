from time import perf_counter
import math
import os
import sys
from pathlib import Path
import numpy as np

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
    beamform_by_dasmtx,
    rf2iq,
    get_fnumber,
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
        workers=None,
    )
    iq_t = rf2iq(signals_rx_t, fc, fs)

    t0 = perf_counter()
    dasmtx = get_dasmtx(
        ula, medium, tx_delays, n_samples, fs, points, fnumber,
        True, fc,

    )
    t1 = perf_counter()
    t_construct_dasmtx = t1-t0

    t0 = perf_counter()
    beamform_by_dasmtx(dasmtx, iq_t)
    t1 = perf_counter()
    t_dasmtx_mul = t1-t0

    print(
        f'Beamforming using DAS matrix: {t_construct_dasmtx+t_dasmtx_mul:.3f} s '
        f'(constructing DAS matrix: {t_construct_dasmtx:.3f} s, '
        f'matrix mul.: {t_dasmtx_mul:.3f} s)'
    )

    t0 = perf_counter()
    beamform(ula, medium, tx_delays, iq_t, fs, points, fnumber, fc)
    t1 = perf_counter()

    t_direct_beamform = t1-t0
    print(f'Beamforming using direct interpolation: {t_direct_beamform:.3f} s')
