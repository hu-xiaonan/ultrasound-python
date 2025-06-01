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
    medium = Medium(rho=1e3, c=1540.0, attenuation=0.5)

    ula_1 = UniformLinearArray(num=64, pitch=300e-6, width=250e-6, height=14e-3)
    props_1 = TransducerProperty(focus=60e-3, baffle=1.75)
    tx_delays_1 = get_tx_delays_from_tilt(ula_1, medium, 0)
    tx_apodization_1 = np.ones(ula_1.num)
    freq_1 = 2.72e6

    ula_2 = UniformLinearArray(num=128, pitch=300e-6, width=270e-6, height=5e-3)
    props_2 = TransducerProperty(focus=18e-3, baffle=1.75)
    tx_delays_2 = get_tx_delays_from_tilt(ula_2, medium, 0)
    tx_apodization_2 = np.ones(ula_2.num)
    freq_2 = 7.6e6

    ula_3 = UniformLinearArray(num=192, pitch=200e-6, width=170e-6, height=5e-3)
    props_3 = TransducerProperty(focus=20e-3, baffle=1.75)
    tx_delays_3 = get_tx_delays_from_tilt(ula_3, medium, 0)
    tx_apodization_3 = np.ones(ula_3.num)
    freq_3 = 7.54e6

    x = np.linspace(-4e-2, 4e-2, 200+1)
    z = np.linspace(0.1e-2, 10e-2, 250+1)
    x_mesh, z_mesh = np.meshgrid(x, z, indexing='xy')
    points = np.transpose([x_mesh.reshape((-1,)), z_mesh.reshape((-1,))])

    titles = [
        'P4-2v (64-element,\n2.7-MHz phased array)',
        'L11-5v (128-element,\n7.6-MHz linear array)',
        'L12-3v (192-element,\n7.5-MHz linear array)',
    ]
    p = [
        get_p_f(ula_1, props_1, medium, tx_delays_1, tx_apodization_1, points, freq_1),
        get_p_f(ula_2, props_2, medium, tx_delays_2, tx_apodization_2, points, freq_2),
        get_p_f(ula_3, props_3, medium, tx_delays_3, tx_apodization_3, points, freq_3),
    ]
    p = np.reshape(p, (-1, *x_mesh.shape))
    p_magnitude = np.abs(p)
    dbl = 20*np.log10(p_magnitude/p_magnitude.max())

    fig, axs = plt.subplots(1, 3, figsize=(9, 4), layout='constrained')
    for i, ula in enumerate([ula_1, ula_2, ula_3]):
        ax = axs[i]
        ax.set_title(titles[i])
        im = ax.imshow(
            dbl[i],
            vmin=-30, vmax=0, cmap='inferno',
            extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
        )
        ax.plot([ula.xmin*100, ula.xmax*100], [0, 0], lw=5, c='C7')
        ax.set_xlabel('$x$ (cm)')
        ax.set_ylabel('$z$ (cm)')
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=axs, pad=0.02, anchor=(0, 0.4), shrink=0.6, ticks=[-30, -20, -10, 0])
    cbar.ax.set_title('dB')

    fig.savefig('fig-S10-probes.png', dpi=300)
