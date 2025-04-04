import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

os.chdir(Path(__file__).parent.resolve())
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from sonic import (
    UniformLinearArray,
    TransducerProperty,
    Medium,
    genscat,
    get_tx_delays_from_tilt,
    get_tx_pulse_t,
    get_signals_rx_t,
)  # noqa: E402


def load_img_and_genscat():
    img = Image.open('heart.jpg')
    img = ImageOps.grayscale(img)
    v = np.asarray(img, dtype=np.float64)/255
    v = v.T
    real_size = 12e-2
    x = np.linspace(-0.5, 0.5, v.shape[0])*real_size
    z = np.arange(v.shape[1])*(x[1]-x[0])+0.1e-2
    bbox = [x.min(), z.min(), x.max(), z.max()]
    rng = np.random.default_rng(202307)
    scatterers, reflectivity = genscat(x, z, v, medium.c/fc, rng)
    return img, bbox, scatterers, reflectivity


if __name__ == '__main__':
    ula = UniformLinearArray(num=64, pitch=300e-6, width=250e-6, height=14e-3)
    props = TransducerProperty(focus=60e-3, baffle=1.75)
    medium = Medium(rho=1e3, c=1540.0, attenuation=0.5)
    fc = 2.72e6
    bandwidth = 0.74
    tx_apodization = np.ones(ula.num)
    n_samples = 4096
    fs = 20e6
    pulse_t = get_tx_pulse_t(n_samples, fs, fc, bandwidth)

    img, bbox, scatterers, reflectivity = load_img_and_genscat()

    # Transmit seven steered plane waves with beam angles evenly spaced from
    # -30 degrees to 30 degrees. The corresponding sets of received signals
    # for these seven angles are referred to as the slow-time dimension.
    tx_delays_arr = [
        get_tx_delays_from_tilt(ula, medium, tilt)
        for tilt in np.deg2rad(np.linspace(-30, 30, 7))
    ]
    signals_rx_t_arr = [
        get_signals_rx_t(
            ula, props, medium, tx_delays, tx_apodization,
            scatterers, reflectivity, pulse_t, fs,
            workers=None, show_pbar=True,
        )
        for tx_delays in tx_delays_arr
    ]
    np.save('data-9-tx_delays.npy', tx_delays_arr)
    np.save('data-9-signals.npy', signals_rx_t_arr)
