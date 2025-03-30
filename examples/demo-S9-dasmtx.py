import math
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

os.chdir(Path(__file__).parent.resolve())
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from sonic import (
    UniformLinearArray,
    Medium,
    get_tx_delays_from_focus,
    get_dasmtx,
    get_fnumber,
)  # noqa: E402

if __name__ == '__main__':
    ula = UniformLinearArray(num=64, pitch=300e-6, width=250e-6, height=14e-3)
    medium = Medium(rho=1e3, c=1540.0, attenuation=0.5)

    tx_delays = get_tx_delays_from_focus(
        ula, medium, [0, -ula.pitch*(ula.num-1)/2/math.tan(math.radians(40))],
    )

    fc = 2.72e6
    bandwidth = 0.74
    fnumber = get_fnumber(ula.width, medium.c/(fc*(1+bandwidth/2)))
    n_samples = 2048
    fs = 10e6

    x = np.linspace(-5e-2, 5e-2, 256)
    z = np.linspace(2e-2, 12e-2, 256)
    x_mesh, z_mesh = np.meshgrid(x, z, indexing='xy')
    points = np.transpose([x_mesh.reshape((-1,)), z_mesh.reshape((-1,))])

    dasmtx = get_dasmtx(ula, medium, tx_delays, n_samples, fs, points, fnumber, True, fc)
    density = dasmtx.nnz/np.prod(dasmtx.shape)
    submtx = dasmtx[
        len(x)*(len(z)//2):len(x)*(len(z)//2+4),
        n_samples*(ula.num//2):n_samples*(ula.num//2+1)
    ]

    fig, axs = plt.subplots(1, 2, figsize=(12, 3), layout='constrained')
    axs[0].set_title(
        f'DAS matrix, size = {dasmtx.shape}, density = {density:.3%}'
    )
    axs[0].spy(dasmtx, marker=',', aspect='equal', c='C2')
    axs[0].tick_params(
        axis='both', which='both',
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=False, labelright=False,
    )
    rect = Rectangle(
        xy=(n_samples*(ula.num//2), len(x)*(len(z)//2)),
        width=n_samples,
        height=4*len(x),
        lw=0.5, edgecolor='C3', fill=False, zorder=999,
    )
    axs[0].add_patch(rect)
    axs[1].set_title(f'Enlarged view, size = {submtx.shape}')
    axs[1].spy(submtx, marker=',', aspect='equal', c='C2')
    for spine in axs[1].spines.values():
        spine.set_color('C3')
    axs[1].tick_params(
        axis='both', which='both',
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labeltop=False, labelleft=False, labelright=False,
    )
    fig.savefig('fig-S9-dasmtx.png', dpi=600)
