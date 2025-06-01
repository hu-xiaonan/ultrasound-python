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
    get_p_f,
)  # noqa: E402

if __name__ == '__main__':
    props = TransducerProperty(focus=math.inf, baffle=math.inf)
    medium = Medium(rho=1e3, c=1540.0, attenuation=0.5)
    tx_del = np.zeros(1)
    tx_apod = np.ones(1)
    freq = 0.5e6
    wavelength = medium.c/freq

    x = np.linspace(-4e-2, 4e-2, 200+1)
    z = np.linspace(0.1e-2, 10e-2, 250+1)
    x_mesh, z_mesh = np.meshgrid(x, z, indexing='xy')
    points = np.transpose([x_mesh.reshape((-1,)), z_mesh.reshape((-1,))])
    points_z_axis = np.transpose(
        [np.zeros(250+1), np.linspace(0.1e-2, 100*wavelength, 250+1)]
    )

    fig, axs = plt.subplots(3, 3, figsize=(9, 9), layout='constrained')
    for i, (width_wavelenghth_ratio, subelem_num) in enumerate(
        [(1, 2), (4, 6), (10, 14)]
    ):
        ula = UniformLinearArray(
            num=1,
            pitch=wavelength*width_wavelenghth_ratio*1.2,
            width=wavelength*width_wavelenghth_ratio,
            height=1e-9,
        )

        p = [
            get_p_f(ula, props, medium, tx_del, tx_apod, points, freq, subelement_num=subelem_num),
            get_p_f(ula, props, medium, tx_del, tx_apod, points, freq, subelement_num=1),
        ]
        p_z_axis = [
            get_p_f(ula, props, medium, tx_del, tx_apod, points_z_axis, freq, subelement_num=subelem_num),
            get_p_f(ula, props, medium, tx_del, tx_apod, points_z_axis, freq, subelement_num=1),
        ]
        p = np.reshape(p, (-1, *x_mesh.shape))
        p_magnitude = np.abs(p)
        dbl = 20*np.log10(p_magnitude/p_magnitude.max())
        p_z_axis_magnitude = np.abs(p_z_axis)
        dbl_z_axis = 20*np.log10(p_z_axis_magnitude/p_z_axis_magnitude.max())

        ax = axs[i, 0]
        ax.set_title(rf'$2b/\lambda={width_wavelenghth_ratio}$'+f'\n(subelements = {subelem_num})')
        im = ax.imshow(
            dbl[0],
            vmin=-100, vmax=0, cmap='inferno',
            extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
        )
        ax.plot([ula.xmin*100, ula.xmax*100], [0, 0], lw=5, c='C7')
        ax.set_xlabel('$x$ (cm)')
        ax.set_ylabel('$z$ (cm)')
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax = axs[i, 1]
        ax.set_title(rf'$2b/\lambda={width_wavelenghth_ratio}$'+'\n(no subelements)')
        im = ax.imshow(
            dbl[1],
            vmin=-100, vmax=0, cmap='inferno',
            extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
        )
        ax.plot([ula.xmin*100, ula.xmax*100], [0, 0], lw=5, c='C7')
        ax.set_xlabel('$x$ (cm)')
        ax.set_ylabel('$z$ (cm)')
        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = fig.colorbar(im, ax=axs[i, 0:2], pad=0.02, anchor=(0, 0.4), shrink=0.6)
        cbar.ax.set_title('dB')

        ax = axs[i, 2]
        ax.set_title(rf'$2b/\lambda={width_wavelenghth_ratio}$')
        ax.plot(points_z_axis[:, 1]/wavelength, dbl_z_axis[0], label=f'subelements = {subelem_num}')
        ax.plot(points_z_axis[:, 1]/wavelength, dbl_z_axis[1], label='no subelements')
        ax.set_xlabel(r'$z/\lambda$')
        ax.set_ylabel(r'dB on $z$-axis')
        ax.legend()

    fig.savefig('fig-D1-subelement_splitting.png', dpi=300)
