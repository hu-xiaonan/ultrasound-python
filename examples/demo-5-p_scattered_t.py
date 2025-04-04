import os
import sys
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

os.chdir(Path(__file__).parent.resolve())
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from sonic import (
    UniformLinearArray,
    TransducerProperty,
    Medium,
    get_tx_delays_from_tilt,
    get_tx_pulse_t,
    get_p_t,
    get_p_scattered_t,
)  # noqa: E402

if __name__ == '__main__':
    ula = UniformLinearArray(num=64, pitch=300e-6, width=250e-6, height=14e-3)
    props = TransducerProperty(focus=60e-3, baffle=1.75)
    medium = Medium(rho=1e3, c=1540.0, attenuation=0.5)
    tx_delays = get_tx_delays_from_tilt(ula, medium, 0)
    tx_apodization = np.ones(ula.num)
    n_samples = 1536
    fs = 10e6
    pulse_t = get_tx_pulse_t(n_samples, fs, fc=2.72e6, bandwidth=0.74)

    x = np.linspace(-4e-2, 4e-2, 160+1)
    z = np.linspace(0.1e-2, 10e-2, 200+1)
    x_mesh, z_mesh = np.meshgrid(x, z, indexing='xy')
    points = np.transpose([x_mesh.reshape((-1,)), z_mesh.reshape((-1,))])

    rng = np.random.default_rng(202307)
    scatterers = rng.uniform([-4e-2, 0.5e-2], [4e-2, 9.5e-2], (50, 2))
    reflectivity = 1e-3*rng.rayleigh(size=100)

    p_primary = get_p_t(
        ula, props, medium, tx_delays, tx_apodization, points, pulse_t, fs,
        workers=None, show_pbar=True,
    )
    p_scattered = get_p_scattered_t(
        ula, props, medium, tx_delays, tx_apodization, points,
        scatterers, 1e-3, pulse_t, fs,
        workers=None, show_pbar=True,
    )
    p = p_primary + p_scattered
    p = p.reshape((*x_mesh.shape, -1))
    p_magnitude = np.abs(p)
    dbls = 20*np.log10(p_magnitude/p_magnitude.max())

    fig, ax = plt.subplots(figsize=(3, 3), layout='constrained')
    im = ax.imshow(
        dbls[:, :, 0],
        vmin=-100, vmax=0, cmap='inferno',
        extent=(x[0]*100, x[-1]*100, z[-1]*100, z[0]*100),
    )
    ax.plot([ula.xmin*100, ula.xmax*100], [0, 0], lw=5, c='C7')
    ax.scatter(scatterers[:, 0]*100, scatterers[:, 1]*100, s=1, c='C0')
    ax.set_xlabel('$x$ (cm)')
    ax.set_ylabel('$z$ (cm)')
    txt = ax.text(
        0.025, 0.02, r't = 0.0 $\mathrm{us}$',
        ha='left', va='bottom', transform=ax.transAxes, color='w',
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, anchor=(0, 0.2), shrink=0.8)
    cbar.ax.set_title('dB')

    fig.canvas.draw()
    fig.set_layout_engine('none')

    def update(i):
        im.set_array(dbls[:, :, i])
        txt.set_text(f'$t$ = {i/fs*1e6:.0f} $\mathrm{{\mu s}}$')

    anim = animation.FuncAnimation(fig, update, frames=range(0, n_samples, 10))
    anim.save('fig-5-p_scattered_t.gif', fps=25, dpi=200)
