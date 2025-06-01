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
    get_tx_delays_from_focus,
    get_p_f,
)  # noqa: E402

if __name__ == '__main__':
    ula = UniformLinearArray(num=64, pitch=300e-6, width=250e-6, height=14e-3)
    props = TransducerProperty(focus=60e-3, baffle=1.75)
    medium = Medium(rho=1e3, c=1540.0, attenuation=0.5)
    freq = 2.72e6
    tx_apodization = np.ones(ula.num)

    titles = [
        'No TX delays',
        'Tilted by 15 degrees',
        'Focus at (2, 3) cm',
        'Virtual focus at (-2, -3) cm',
    ]
    tx_delays = [
        get_tx_delays_from_tilt(ula, medium, 0),
        get_tx_delays_from_tilt(ula, medium, math.radians(15)),
        get_tx_delays_from_focus(ula, medium, [2e-2, 3e-2]),
        get_tx_delays_from_focus(ula, medium, [-2e-2, -3e-2]),
    ]

    x = np.linspace(-4e-2, 4e-2, 200+1)
    z = np.linspace(0.1e-2, 10e-2, 250+1)
    x_mesh, z_mesh = np.meshgrid(x, z, indexing='xy')
    points = np.transpose([x_mesh.reshape((-1,)), z_mesh.reshape((-1,))])

    p = [
        get_p_f(ula, props, medium, tx_del, tx_apodization, points, freq)
        for tx_del in tx_delays
    ]
    p = np.reshape(p, (-1, *x_mesh.shape))
    p_magnitude = np.abs(p)
    dbl = 20*np.log10(p_magnitude/p_magnitude.max())

    fig, axs = plt.subplots(2, 4, figsize=(12, 5), height_ratios=[1, 4], layout='constrained')
    for i in range(4):
        ax1 = axs[0, i]
        ax1.set_title(titles[i])
        markerline, stemlines, baseline = ax1.stem(
            range(1, 64+1),
            tx_delays[i]*1e6,
            linefmt='k-',
            basefmt='',
        )
        markerline.set_markersize(1)
        stemlines.set_linewidth(0.5)
        baseline.set_linewidth(0.5)
        ax1.set_xlabel('Element #')
        ax1.set_ylabel(r'TX delay ($\mathrm{\mu s}$)')
        ax1.set_xticks([1, 64])
        ax1.set_ylim(0, 8.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax2 = axs[1, i]
        im = ax2.imshow(
            dbl[i],
            vmin=-20, vmax=0, cmap='inferno',
            extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
        )
        ax2.plot([ula.xmin*100, ula.xmax*100], [0, 0], lw=5, c='C7')
        ax2.set_xlabel('$x$ (cm)')
        ax2.set_ylabel('$z$ (cm)')
        for spine in ax2.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axs[1:], pad=0.02, anchor=(0, 0.4), shrink=0.6, ticks=[-20, -10, 0])
    cbar.ax.set_title('dB')

    fig.savefig('fig-1-phased_array.png', dpi=300)
