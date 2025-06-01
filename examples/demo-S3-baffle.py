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
    get_tx_delays_from_tilt,
    get_p_f,
)  # noqa: E402

if __name__ == '__main__':
    ula = UniformLinearArray(num=64, pitch=300e-6, width=250e-6, height=14e-3)
    medium = Medium(rho=1e3, c=1540.0, attenuation=0.5)
    tx_delays = get_tx_delays_from_tilt(ula, medium, 0)
    tx_apodization = np.ones(ula.num)
    freq = 2.72e6

    props_rigid_baffle = TransducerProperty(focus=60e-3, baffle=0)
    props_finite_impedance_baffle = TransducerProperty(focus=60e-3, baffle=1.75)
    props_soft_baffle = TransducerProperty(focus=60e-3, baffle=np.inf)

    x = np.linspace(-4e-2, 4e-2, 200+1)
    z = np.linspace(0.1e-2, 10e-2, 250+1)
    x_mesh, z_mesh = np.meshgrid(x, z, indexing='xy')
    points = np.transpose([x_mesh.reshape((-1,)), z_mesh.reshape((-1,))])

    titles = [
        'Planar wave\n(rigid baffle)',
        'Planar wave\n(impedance ratio = 1.75)',
        'Planar wave\n(soft baffle)',
    ]
    p = [
        get_p_f(ula, props_rigid_baffle, medium, tx_delays, tx_apodization, points, freq),
        get_p_f(ula, props_finite_impedance_baffle, medium, tx_delays, tx_apodization, points, freq),
        get_p_f(ula, props_soft_baffle, medium, tx_delays, tx_apodization, points, freq),
    ]
    p = np.reshape(p, (-1, *x_mesh.shape))
    p_magnitude = np.abs(p)
    # Normalize each field independently.
    dbl = 20*np.log10(p_magnitude/p_magnitude.max(axis=(1, 2), keepdims=True))

    fig, axs = plt.subplots(1, 3, figsize=(9, 4), layout='constrained')
    for i in range(3):
        ax = axs[i]
        ax.set_title(titles[i])
        im = ax.imshow(
            dbl[i],
            vmin=-20, vmax=0, cmap='inferno',
            extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
        )
        ax.plot([ula.xmin*100, ula.xmax*100], [0, 0], lw=5, c='C7')
        ax.set_xlabel('$x$ (cm)')
        ax.set_ylabel('$z$ (cm)')
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axs, pad=0.02, anchor=(0, 0.4), shrink=0.6, ticks=[-20, -10, 0])
    cbar.ax.set_title('dB')

    fig.savefig('fig-S3-baffle.png', dpi=300)
