import os
import sys
from pathlib import Path
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

os.chdir(Path(__file__).parent.resolve())
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from sonic import (
    UniformLinearArray,
    TransducerProperty,
    Medium,
    genscat,
    rf2iq,
    beamform,
    log_compress,
    get_fnumber,
    time_gain_compensate,
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
    fnumber = get_fnumber(ula.width, medium.c/(fc*(1+bandwidth/2)))
    n_samples = 4096
    fs = 20e6
    time = np.linspace(0, n_samples/fs, n_samples)

    img, bbox, scatterers, reflectivity = load_img_and_genscat()

    xmin, zmin, xmax, zmax = bbox
    x = np.linspace(xmin, xmax, 256)
    z = np.linspace(zmin+0.5e-2, zmax, 256)
    x_mesh, z_mesh = np.meshgrid(x, z, indexing='xy')
    points = np.transpose([x_mesh.reshape((-1,)), z_mesh.reshape((-1,))])

    tx_delays_arr = np.load('data-9-tx_delays.npy')
    signals_rx_t_arr = np.load('data-9-signals.npy')
    assert(tx_delays_arr.shape[0] == signals_rx_t_arr.shape[0])

    iq_t_arr = rf2iq(signals_rx_t_arr, fc, fs)
    iq_tgc_t_arr = time_gain_compensate(iq_t_arr)
    beamformed = [
        beamform(ula, medium, tx_delays, compensated_t, fs, points, fnumber, fc)
        for tx_delays, compensated_t in zip(tx_delays_arr, iq_tgc_t_arr)
    ]
    beamformed = np.reshape(beamformed, (-1, *x_mesh.shape))
    echo_img = np.sum(beamformed, axis=0)

    dynamic_range = 50
    echo_img = log_compress(echo_img, dynamic_range=dynamic_range)

    fig = plt.figure(figsize=(10, 8), layout='tight')
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Reference image')
    ax1.imshow(img, cmap='gray')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Generated scatterers')
    ax2.scatter(*(scatterers*1e2).T, s=reflectivity)
    ax2.set_xlim(xmin*1e2, xmax*1e2)
    ax2.set_ylim(zmax*1e2, zmin*1e2)
    ax2.set_xlabel('$x$ (cm)')
    ax2.set_ylabel('$z$ (cm)')
    ax2.set_aspect('equal')

    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    ax3.set_title('Received signals')
    ax3.view_init(elev=30, azim=210)
    signals_normalized_arr = signals_rx_t_arr/np.ptp(signals_rx_t_arr)
    for slow_time in range(signals_rx_t_arr.shape[0]):
        for i in range(signals_rx_t_arr.shape[1]):
            ax3.plot(
                np.full_like(time, slow_time+1),
                signals_normalized_arr[slow_time, i]+i+1,
                time*1e6,
                zorder=-slow_time,
                lw=0.5, c=plt.colormaps['tab10'].colors[slow_time],
            )
    ax3.set_xlabel('Slow-time', labelpad=-3)
    ax3.set_ylabel('Element #', labelpad=-3)
    ax3.set_yticks([1, 16, 32, 48, 64])
    ax3.set_zlabel(r'Fast-time ($\mathrm{\mu s}$)', labelpad=-3)
    ax3.invert_yaxis()
    ax3.invert_zaxis()
    # https://stackoverflow.com/questions/62185161/move-the-z-axis-on-the-other-side-on-a-3d-plot-python
    # tmp_planes = ax3.zaxis._PLANES
    # ax3.zaxis._PLANES = (
    #     tmp_planes[2], tmp_planes[3], 
    #     tmp_planes[0], tmp_planes[1], 
    #     tmp_planes[4], tmp_planes[5],
    # )
    ax3.tick_params(axis='both', which='major', pad=-1)
    ax3.tick_params(axis='both', which='minor', pad=-1)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title('B-mode echo image')
    im = ax4.imshow(
        echo_img,
        vmin=-dynamic_range, vmax=0, cmap='gray',
        extent=(xmin*1e2, xmax*1e2, zmax*1e2, zmin*1e2),
    )
    ax4.set_xlabel('$x$ (cm)')
    ax4.set_ylabel('$z$ (cm)')
    for spine in ax4.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax4)
    cbar.ax.set_title('dB')

    # https://stackoverflow.com/questions/21918380/rotating-axes-label-text-in-3d
    fig.canvas.draw()
    ax3.zaxis.set_rotate_label(False)
    ax3.zaxis.label.set_rotation((ax3.zaxis.label.get_rotation()+180)%360)

    fig.savefig('fig-9-echocardiography.png', dpi=300)
