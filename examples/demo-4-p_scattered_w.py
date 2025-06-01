# Copyright (c) 2023-2025 Hu Xiaonan
# License: MIT License

import os
import sys
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

os.chdir(Path(__file__).parent.resolve())
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from sonic import (
    UniformLinearArray,
    TransducerProperty,
    Medium,
    get_tx_delays_from_focus,
    get_p_f,
    get_p_scattered_f,
)  # noqa: E402

if __name__ == '__main__':
    ula = UniformLinearArray(num=64, pitch=300e-6, width=250e-6, height=14e-3)
    props = TransducerProperty(focus=60e-3, baffle=1.75)
    medium = Medium(rho=1e3, c=1540.0, attenuation=0.5)
    tx_delays = get_tx_delays_from_focus(ula, medium, [-1e-2, -2e-2])
    tx_apodization = np.ones(ula.num)
    freq = 2.72e6

    x = np.linspace(-4e-2, 4e-2, 200+1)
    z = np.linspace(0.1e-2, 10e-2, 250+1)
    x_mesh, z_mesh = np.meshgrid(x, z, indexing='xy')
    points = np.transpose([x_mesh.reshape((-1,)), z_mesh.reshape((-1,))])

    rng = np.random.default_rng(202307)
    scatterers = rng.uniform([-4e-2, 0.5e-2], [4e-2, 10e-2], (100, 2))
    reflectivity = 1e-3*rng.rayleigh(size=100)

    p_primary = get_p_f(ula, props, medium, tx_delays, tx_apodization, points, freq)
    p_scattered = get_p_scattered_f(
        ula, props, medium, tx_delays, tx_apodization,
        points, scatterers, reflectivity, freq,
    )
    titles = ['Primary', 'Scattered', 'Total']
    p = [p_primary, p_scattered, p_primary + p_scattered]
    p = np.reshape(p, (-1, *x_mesh.shape))
    p_magnitude = np.abs(p)
    dbl = 20*np.log10(p_magnitude/p_magnitude.max())

    fig, axs = plt.subplots(1, 3, figsize=(9, 4), layout='constrained')
    for i in range(3):
        ax = axs[i]
        ax.set_title(titles[i])
        im = ax.imshow(
            dbl[i],
            vmin=-60, vmax=0, cmap='inferno',
            extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
        )
        ax.plot([ula.xmin*100, ula.xmax*100], [0, 0], lw=5, c='C7')
        ax.set_xlabel('$x$ (cm)')
        ax.set_ylabel('$z$ (cm)')
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axs, pad=0.02, anchor=(0, 0.4), shrink=0.6)
    cbar.ax.set_title('dB')

    fig.savefig('fig-4-p_scattered_w.png', dpi=300)
