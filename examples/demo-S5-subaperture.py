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
    subaperture = np.asarray([1.0]*16+[0.0]*(64-16))

    tx_delays_planar = get_tx_delays_from_tilt(ula, medium, 0)
    tx_delays_focused = get_tx_delays_from_focus(ula, medium, [0e-2, 6e-2])

    x = np.linspace(-4e-2, 4e-2, 200+1)
    z = np.linspace(0.1e-2, 10e-2, 250+1)
    x_mesh, z_mesh = np.meshgrid(x, z, indexing='xy')
    points = np.transpose([x_mesh.reshape((-1,)), z_mesh.reshape((-1,))])

    titles = [
        'Planar wave\nwith elements #1 to #16',
        'Focused wave\nwith elements #1 to #16',
    ]
    p = [
        get_p_f(ula, props, medium, tx_delays_planar, subaperture, points, freq),
        get_p_f(ula, props, medium, tx_delays_focused, subaperture, points, freq),
    ]
    p = np.reshape(p, (-1, *x_mesh.shape))
    p_magnitude = np.abs(p)
    dbl = 20*np.log10(p_magnitude/p_magnitude.max())

    fig, axs = plt.subplots(1, 2, figsize=(6, 4), layout='constrained')
    for i in range(2):
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

    fig.savefig('fig-S5-subaperture.png', dpi=300)
