from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import math
import numpy as np
from scipy.interpolate import interpn
from scipy.optimize import minimize_scalar
from scipy.signal import butter, filtfilt, hilbert
from scipy.sparse import csr_array
from scipy.spatial.distance import cdist
from scipy.special import erf
from tqdm import tqdm


def _pulse_w(freqs, fc, n_lambda):
    """
    Return the transmit wave.

    Parameters
    ----------
    n_samples : array_like
        Number of samples.
    sampling_rate : float
        Sampling rate.
    fc : float
        Center frequency of the transducer.
    bandwidth : float
        Bandwidth of the transducer.
    n_lambda : int, optional
        Number of sine wave cycles, default is 1.

    Returns
    -------
    pulse_t : numpy.ndarray, shape=(n_samples,)
        The transmit wave.

    """
    return 1j*(np.sinc(n_lambda*(freqs-fc)/fc)-np.sinc(n_lambda*(freqs+fc)/fc))


def _probe_w(freqs, fc, bandwidth):
    """
    Return the spectrum of the transducer Point Spread Function (PSF).

    Parameters
    ----------
    freqs : array_like
        Frequencies to calculate.
    fc : float
        Center frequency of the transducer.
    bandwidth : float
        Bandwidth of the transducer.

    Returns
    -------
    spectrum : numpy.ndarray, shape=freqs.shape
        The spectrum of the transducer PSF.

    """
    fb = fc*bandwidth
    p = math.log(126)/math.log(2/bandwidth)
    return np.exp2(-(2*np.abs(freqs-fc)/fb)**p)


def get_tx_pulse_t(n_samples, sampling_rate, fc, bandwidth, n_lambda=1):
    """
    Return the transmit wave.

    Parameters
    ----------
    n_samples : array_like
        Number of samples.
    sampling_rate : float
        Sampling rate.
    fc : float
        Center frequency of the transducer.
    bandwidth : float
        Bandwidth of the transducer.
    n_lambda : int, optional
        Number of sine wave cycles, default is 1.

    Returns
    -------
    pulse_t : numpy.ndarray, shape=(n_samples,)
        The transmit wave.

    """
    freqs = np.fft.fftfreq(n_samples, 1/sampling_rate)
    return np.fft.ifft(
        _pulse_w(freqs, fc, n_lambda)*_probe_w(freqs, fc, bandwidth)
    ).real


class UniformLinearArray:
    def __init__(self, num, pitch, width, height):
        self.num = num
        self.pitch = pitch
        self.width = width
        self.height = height
        self.xmax = (num-1)*pitch/2+width/2
        self.xmin = -self.xmax
        xcmax = pitch*(num-1)/2
        self.xc = np.linspace(-xcmax, xcmax, num)
        self.zc = np.zeros(num)
        self.pc = np.transpose([self.xc, self.zc])
        self.tilt = np.zeros(num)

    def subelement_split(self, num):
        w = self.width
        offset = np.linspace(-w/2*(1-1/num), w/2*(1-1/num), num)
        subelement_xc = self.xc[:, np.newaxis]+offset
        subelement_zc = np.zeros_like(subelement_xc)
        subelement_tilt = np.zeros_like(subelement_xc)
        return (subelement_xc, subelement_zc, subelement_tilt)


class TransducerProperty:
    def __init__(self, focus, baffle):
        self.focus = focus  # Elevation focus.
        self.baffle = baffle  # Ratio of the medium and baffled impedance.


class Medium:
    def __init__(self, rho, c, attenuation):
        self.rho = rho
        self.c = c
        self.attenuation = attenuation


def get_tx_delays_from_tilt(ula, medium, tilt):
    """
    Return the transmit time delays required to form a tilted plane wave.

    Parameters
    ----------
    ula : UniformLinearArray
    medium : Medium
    tilt : float
        Tilt angle (in radians) with respect to the z-axis. If zero, returns an
        array of zeros, which corresponds to the wave along the z-axis.

    Returns
    -------
    delays : numpy.ndarray, shape=(ula.num,)
        Transmit time delays.

    """
    delays = ula.xc/medium.c*math.sin(tilt)
    return delays-np.min(delays)


def get_tx_delays_from_focus(ula, medium, focus):
    """
    Returns the transmit time delay required to form a focused wave.

    Parameters
    ----------
    ula : UniformLinearArray
    medium : Medium
    focus : array_like, shape=(2,)
        Focus of the wave in coordinate (x, z). If z is negative, then the
        focus is virtual, corresponding to a circular wave.

    Returns
    -------
    delays : numpy.ndarray, shape=(ula.num,)
        Transmit time delays.

    """
    distances = np.hypot(ula.xc-focus[0], ula.zc-focus[1])
    delays = -distances/medium.c if focus[1] > 0 else distances/medium.c
    return delays-np.min(delays)


def _get_geometry_results(ula, points, subelement_num):
    """
    Return the geometric quantities required for pressure field calculation.

    Parameters
    ----------
    ula : UniformLinearArray
    points : array_like, shape=(n, 2)
        Locations at which to calculate pressure.
    subelement_num : int
        Number of sub-elements.

    Returns
    -------
    r : numpy.ndarray, shape=(len(points), ula.num, subelement_num)
        r[i, j, k] is the distance between points[i] and the center of the k-th
        sub-element of the j-th element of the ULA, .
    angles : numpy.ndarray, shape=r.shape
        angles[i, j, k] is the directed angle of point[i] to the center of the
        k-th sub-element of the j-th element of the ULA.
    sin_angles : numpy.ndarray
        sin_angles = sin(angles).
    cos_angles : numpy.ndarray
        cos_angles = cos(angles).

    Notes
    -----
    Since geometry quantities are frequency independent, they are calculated in
    advance to avoid redundant calculations in loops.

    The calculation of pressure field is based on far-field condition. If the
    width of the ULA elements is close or larger than the wave length,
    sub-element splitting should be used to ensure that the far-field condition
    is met.

    """
    x = points[:, 0]
    z = points[:, 1]
    (
        subelement_xc,
        subelement_zc,
        subelement_tilt,
    ) = ula.subelement_split(subelement_num)
    relative_x = x[:, np.newaxis, np.newaxis]-subelement_xc
    relative_z = z[:, np.newaxis, np.newaxis]-subelement_zc
    r = np.hypot(relative_x, relative_z)
    angles = np.arctan2(relative_x, relative_z)-subelement_tilt
    return (r, angles, np.sin(angles), np.cos(angles))


def get_fnumber(width, wave_length):
    """
    Return the receive f-number of the element.

    The formula used in this function is based on Eq. (13) from the following
    paper:

    Perrot, V., Polichetti, M., Varray, F., & Garcia, D. (2021). So you think
    you can DAS? A viewpoint on delay-and-sum beamforming. Ultrasonics, 111,
    106309.

    https://doi.org/10.1016/j.ultras.2020.106309

    Parameters
    ----------
    width : float
        Element width.
    wave_length : float
        Wave length of the transmit signal.

    Returns
    -------
    fnumber : float
        The receive f-number.

    """
    def f(th):
        return abs(math.cos(th)*np.sinc(width/wave_length*math.sin(th))-0.71)

    alpha = minimize_scalar(f, bounds=(0, math.pi/2)).x
    return 1/(2*math.tan(alpha))


def _get_integral_over_h(Rf, h, k, r):
    """
    Compute the integral in Eq. (11) from the
    [SIMUS](https://doi.org/10.1016/j.cmpb.2022.106726) paper using the
    Gaussian error function.

    The formula used in this function was derived with the assistance of
    Wolfram Alpha, accessible via the following query:

    https://www.wolframalpha.com/input?i=int+exp%28-i*a*y%5E2%29+from+y+%3D+-h%2F2+to+h%2F2

    Parameters
    ----------
    Rf : float
        Elevation focus.
    h : float
        Height of the ULA elements.
    k : float
        Wave number.
    r : numpy.ndarray
        Distances.

    Returns
    -------
    integral : numpy.ndarray, shape=r.shape
        The integral required for pressure field calculations.

    """
    sqrt_pi = math.sqrt(math.pi)

    a = np.complex128(k/2*(1/Rf-1/r))
    mask = a != 0
    sqrt_a_masked = np.sqrt(a[mask])
    result = np.empty_like(a)
    result[mask] = -(
        (-1)**(3/4)*sqrt_pi*erf(1/2*(-1)**(1/4)*h*sqrt_a_masked)
    )/sqrt_a_masked
    result[~mask] = h
    return result


def _get_p_f(
    ula, props, medium, tx_delays, tx_apodization,
    precalculated_geometry, f,
):
    """
    Target function for parallel computing of frequency response of pressure.

    The formula used in this function is based on the following paper:

    Garcia, D. (2022). SIMUS: An open-source simulator for medical ultrasound
    imaging. Part I: Theory & examples. Computer Methods and Programs in
    Biomedicine, 218, 106726.

    https://doi.org/10.1016/j.cmpb.2022.106726

    Parameters
    ----------
    ula : UniformLinearArray
    props : TransducerProperty
    medium : Medium
    tx_delays : numpy.ndarray, shape=(ula.num,)
        Transmit time delays.
    tx_apodization : numpy.ndarray, shape=(ula.num,)
        Transmit apodization.
    precalculated_geometry : tuple
        A tuple of geometric quantities, in the form of (r, angles, sin_angles,
        cos_angles). For each element in the tuple, type = numpy.ndarray and
        shape=(len(points), ula.num, subelement_num). For more, see the
        documentation of `_get_geometry_results`.
    f : float
        Frequency.

    Returns
    -------
    p : numpy.ndarray, shape=(len(points),)
        The complex pressure field, where p[i] is the pressure of points[i]
        under the given frequency.

    """
    (r, angles, sin_angles, cos_angles) = precalculated_geometry
    w = math.tau*f
    k = w/medium.c
    vz = np.exp(-1j*w*tx_delays)*tx_apodization
    ka = medium.attenuation/8.69*abs(f)/1e6*1e2
    p = -1j*medium.rho*f*vz[:, np.newaxis]*np.exp(-(ka+1j*k)*r)/r
    integral_over_b = ula.width*np.sinc(1/math.pi*k*ula.width/2*sin_angles)
    integral_over_h = _get_integral_over_h(props.focus, ula.height, k, r)
    p *= integral_over_b
    p *= integral_over_h
    # Finite impedance baffle.
    if math.isinf(props.baffle):
        # Soft baffle.
        p *= cos_angles
    elif props.baffle > 0.0:
        # Finite impedance baffle.
        p *= cos_angles/(cos_angles+props.baffle)
    else:
        # Rigid Baffle:
        pass
    return p.sum(axis=2).sum(axis=1)


def get_p_f(
    ula, props, medium, tx_delays, tx_apodization, points, f,
    subelement_num=None,
):
    """
    Return the pressure generated by the ULA at the given frequency.

    Parameters
    ----------
    ula : UniformLinearArray
    props : TransducerProperty
    medium : Medium
    tx_delays : array_like, shape=(ula.num,).
        Transmit time delays.
    tx_apodization : array_like, shape=(ula.num,)
        Transmit apodization.
    points : array_like, shape=(n, 2)
        Locations at which to calculate pressure.
    f : float
        Frequency.
    subelement_num : int or None, optional
        The number of sub-elements. Default is None, which automatically
        calculates the number of sub-elements.

    Returns
    -------
    p : numpy.ndarray, shape=(len(points),)
        The complex pressure field, where p[i] is the pressure of points[i]
        under the given frequency.

    """
    tx_delays = np.asarray(tx_delays)
    tx_apodization = np.asarray(tx_apodization)
    if subelement_num is None:
        wave_length = medium.c/f
        subelement_num = math.ceil(ula.width/wave_length)
    precalculated_geometry = _get_geometry_results(ula, points, subelement_num)
    return _get_p_f(
        ula, props, medium, tx_delays, tx_apodization,
        precalculated_geometry, f,
    )


def get_p_f_vectorized(
    ula, props, medium, tx_delays, tx_apodization, points, freqs,
    subelement_num=None, *,
    workers=1, show_pbar=False,
):
    """
    Vectorized version of `get_p_f`.

    Parameters
    ----------
    ula : UniformLinearArray
    props : TransducerProperty
    medium : Medium
    tx_delays : array_like, shape=(ula.num,).
        Transmit time delays.
    tx_apodization : array_like, shape=(ula.num,)
        Transmit apodization.
    points : array_like, shape=(n, 2)
        Locations at which to calculate pressure.
    freqs : array_like
        The frequencies.
    subelement_num : int or None, optional
        The number of sub-elements. Default is None, which automatically
        calculates the number of sub-elements.
    workers : int or any type, keyword-only, optional
        If `workers != 1`, then `concurrent.futures.ProcessPoolExecutor` is
        used and `workers` is passed as `max_workers` to
        `concurrent.futures.ProcessPoolExecutor`. The default is 1, that
        parallel computing is not used.
    show_pbar : bool, keyword-only, optional
        If True, show a progress bar. Default is False.

    Returns
    -------
    p : numpy.ndarray, shape=(len(freqs), len(points)).
        The complex pressure field, where p[i, j] is the pressure of points[j]
        under freqs[i].

    """
    tx_delays = np.asarray(tx_delays)
    tx_apodization = np.asarray(tx_apodization)
    if subelement_num is None:
        wave_length = medium.c/freqs.max()
        subelement_num = math.ceil(ula.width/wave_length)
    precalculated_geometry = _get_geometry_results(ula, points, subelement_num)
    p_f_arr = []
    if show_pbar:
        print('Computing the pressure field generated by the transducer...')
    if workers == 1:
        iterator = tqdm(freqs) if show_pbar else freqs
        for f in iterator:
            p_f_arr.append(
                _get_p_f(
                    ula, props, medium, tx_delays, tx_apodization,
                    precalculated_geometry, f,
                )
            )
    else:
        with ProcessPoolExecutor(workers) as executor:
            iterator = executor.map(
                _get_p_f,
                repeat(ula),
                repeat(props),
                repeat(medium),
                repeat(tx_delays),
                repeat(tx_apodization),
                repeat(precalculated_geometry),
                freqs,
            )
            if show_pbar:
                iterator = tqdm(iterator, total=len(freqs))
            for p_f in iterator:
                p_f_arr.append(p_f)
    return np.asarray(p_f_arr)


def get_p_t(
    ula, props, medium, tx_delays, tx_apodization, points,
    pulse_t, sampling_rate,
    subelement_num=None, db_thresh=-math.inf, *,
    workers=1, show_pbar=False,
):
    """
    Return the pressure generated by the ULA versus time.

    Parameters
    ----------
    ula : UniformLinearArray
    props : TransducerProperty
    medium : Medium
    tx_delays : array_like, shape=(ula.num,).
        Transmit time delays.
    tx_apodization : array_like, shape=(ula.num,)
        Transmit apodization.
    points : array_like, shape=(n, 2)
        Locations at which to calculate pressure.
    pulse_t : array_like, shape=(n_samples,)
        Pulse emitted by the ULA, in time domain.
    sampling_rate : float
        Sampling rate.
    subelement_num : int or None, optional
        The number of sub-elements. Default is None, which automatically
        calculates the number of sub-elements.
    db_thresh : float, optional
        The frequency components whose amplitude is below db_thresh are ignored
        to improve performance.
    workers : int or any type, keyword-only, optional
        If `workers != 1`, then `concurrent.futures.ProcessPoolExecutor` is
        used and `workers` is passed as `max_workers` to
        `concurrent.futures.ProcessPoolExecutor`. The default is 1, that
        parallel computing is not used.
    show_pbar : bool, keyword-only, optional
        If True, show a progress bar. Default is False.

    Returns
    -------
    p : numpy.ndarray, shape=(len(points), len(pulse_t))
        The complex pressure field, where p[i, j] is the pressure of points[i]
        at time[j].

    """
    n_samples = len(pulse_t)
    freqs = np.fft.fftfreq(n_samples, 1/sampling_rate)
    pulse_w = np.fft.fft(pulse_t)
    # Ignore insignificant frequencies.
    pulse_w_mag = np.abs(pulse_w)
    idx = pulse_w_mag >= 10**(db_thresh/20)*pulse_w_mag.max()
    p_f_arr = np.zeros((n_samples, len(points)), dtype=np.complex128)
    p_f_arr[idx] = get_p_f_vectorized(
        ula, props, medium, tx_delays, tx_apodization, points, freqs[idx],
        subelement_num=subelement_num, workers=workers, show_pbar=show_pbar,
    )
    return np.fft.ifft(pulse_w*p_f_arr.T, axis=1)


def _get_p_scattered_f(medium, intensity, r, f):
    """
    Target function for parallel computing of frequency response of scattered
    pressure.

    Parameters
    ----------
    medium : Medium
    intensity : complex numpy.ndarray, shape=(len(scatterers),)
        The pressure at the scatterers.
    r : numpy.ndarray
        The distances from the points to the scatters.
    f : float
        Frequency.

    Returns
    -------
    p: complex numpy.ndarray, shape=r.shape
        The pressure field, where p[i] is the scattered pressure of points[i]
        under the given frequency.

    """
    w = math.tau*f
    k = w/medium.c
    ka = medium.attenuation/8.69*abs(f)/1e6*1e2
    p = intensity*np.exp(-(ka+1j*k)*r)/r
    return p.sum(axis=1)


def get_p_scattered_f(
    ula, props, medium, tx_delays, tx_apodization,
    points, scatterers, reflectivity, f,
    subelement_num=None,
):
    """
    Return the scattered pressure at the given frequency.

    Parameters
    ----------
    ula : UniformLinearArray
    props : TransducerProperty
    medium : Medium
    tx_delays : array_like, shape=(ula.num,).
        Transmit time delays.
    tx_apodization : array_like, shape=(ula.num,)
        Transmit apodization.
    points : array_like, shape=(n, 2)
        Locations at which to calculate pressure.
    scatterers : array_like, shape=(m, 2)
        Locations of the scatterers.
    reflectivity : array_like, shape=(m,)
        The reflectivity of the scatterers.
    f : float
        Frequency.
    subelement_num : int or None, optional
        The number of sub-elements. Default is None, which automatically
        calculates the number of sub-elements.

    Returns
    -------
    p : complex numpy.ndarray, shape=(len(points),)
        The complex pressure field, where p[i] is the scattered pressure of
        points[i] under the given frequency.

    """
    intensity = -get_p_f(
        ula, props, medium, tx_delays, tx_apodization, scatterers, f,
        subelement_num=subelement_num,
    )*reflectivity
    r = cdist(points, scatterers)
    return _get_p_scattered_f(medium, intensity, r, f)


def get_p_scattered_f_vectorized(
    ula, props, medium, tx_delays, tx_apodization,
    points, scatterers, reflectivity, freqs,
    subelement_num=None, *,
    workers=1, show_pbar=False,
):
    """
    Vectorized version of `get_p_scattered_f`.

    Parameters
    ----------
    ula : UniformLinearArray
    props : TransducerProperty
    medium : Medium
    tx_delays : array_like, shape=(ula.num,).
        Transmit time delays.
    tx_apodization : array_like, shape=(ula.num,)
        Transmit apodization.
    points : array_like, shape=(n, 2)
        Locations at which to calculate pressure.
    scatterers : array_like, shape=(m, 2)
        Locations of the scatterers.
    reflectivity : array_like, shape=(m,)
        The reflectivity of the scatterers.
    freqs : array_like
        The frequencies.
    subelement_num : int or None, optional
        The number of sub-elements. Default is None, which automatically
        calculates the number of sub-elements.
    workers : int or any type, keyword-only, optional
        If `workers != 1`, then `concurrent.futures.ProcessPoolExecutor` is
        used and `workers` is passed as `max_workers` to
        `concurrent.futures.ProcessPoolExecutor`. The default is 1, that
        parallel computing is not used.
    show_pbar : bool, keyword-only, optional
        If True, show a progress bar. Default is False.

    Returns
    -------
    p : complex numpy.ndarray, shape=(len(freqs), len(points))
        The complex pressure field, where p[i, j] is the scattered pressure of
        points[j] under freqs[i].

    """
    intensity = -get_p_f_vectorized(
        ula, props, medium, tx_delays, tx_apodization, scatterers, freqs,
        subelement_num=subelement_num, workers=workers, show_pbar=show_pbar,
    )*reflectivity
    r = cdist(points, scatterers)
    p_scattered_f_arr = []
    if show_pbar:
        print('Computing the scattered pressure field...')
    if workers == 1:
        iterator = tqdm(range(len(freqs))) if show_pbar else range(len(freqs))
        for i in iterator:
            p_scattered_f_arr.append(
                _get_p_scattered_f(medium, intensity[i], r, freqs[i])
            )
    else:
        with ProcessPoolExecutor(workers) as executor:
            iterator = executor.map(
                _get_p_scattered_f,
                repeat(medium),
                intensity,
                repeat(r),
                freqs,
            )
            if show_pbar:
                iterator = tqdm(iterator, total=len(freqs))
            for p_scattered_f in iterator:
                p_scattered_f_arr.append(p_scattered_f)
    return np.asarray(p_scattered_f_arr)


def get_p_scattered_t(
    ula, props, medium, tx_delays, tx_apodization,
    points, scatterers, reflectivity, pulse_t, sampling_rate,
    subelement_num=None, db_thresh=-math.inf, *,
    workers=1, show_pbar=False,
):
    """
    Return the scattered pressure versus time.

    Parameters
    ----------
    ula : UniformLinearArray
    props : TransducerProperty
    medium : Medium
    tx_delays : array_like, shape=(ula.num,).
        Transmit time delays.
    tx_apodization : array_like, shape=(ula.num,)
        Transmit apodization.
    points : array_like, shape=(n, 2)
        Locations at which to calculate pressure.
    scatterers : array_like, shape=(m, 2)
        Locations of the scatterers.
    reflectivity : array_like, shape=(m,)
        The reflectivity of the scatterers.
    pulse_t : array_like, shape=(n_samples,)
        Pulse emitted by the ULA, in time domain.
    sampling_rate : float
        Sampling rate.
    subelement_num : int or None, optional
        The number of sub-elements. Default is None, which automatically
        calculates the number of sub-elements.
    db_thresh : float, optional
        The frequency components whose amplitude is below db_thresh are ignored
        to improve performance.
    workers : int or any type, keyword-only, optional
        If `workers != 1`, then `concurrent.futures.ProcessPoolExecutor` is
        used and `workers` is passed as `max_workers` to
        `concurrent.futures.ProcessPoolExecutor`. The default is 1, that
        parallel computing is not used.
    show_pbar : bool, keyword-only, optional
        If True, show a progress bar. Default is False.

    Returns
    -------
    p : numpy.ndarray, shape=(len(points), len(pulse_t))
        The complex pressure field, where p[i, j] is the scattered pressure of
        points[i] at time[j].

    """
    n_samples = len(pulse_t)
    freqs = np.fft.fftfreq(n_samples, 1/sampling_rate)
    pulse_w = np.fft.fft(pulse_t)
    # Ignore insignificant frequencies.
    pulse_w_mag = np.abs(pulse_w)
    idx = pulse_w_mag >= 10**(db_thresh/20)*pulse_w_mag.max()
    p_scattered_f_arr = np.zeros((n_samples, len(points)), dtype=np.complex128)
    p_scattered_f_arr[idx] = get_p_scattered_f_vectorized(
        ula, props, medium, tx_delays, tx_apodization,
        points, scatterers, reflectivity, freqs[idx],
        subelement_num=subelement_num, workers=workers, show_pbar=show_pbar,
    )
    return np.fft.ifft(pulse_w*p_scattered_f_arr.T, axis=1)


def get_signals_rx_t(
    ula, props, medium, tx_delays, tx_apodization,
    scatterers, reflectivity, pulse_t, sampling_rate,
    subelement_num=None, db_thresh=-math.inf, *,
    workers=1, show_pbar=False,
):
    """
    Return the simulated RF signal received by the ULA elements.

    Parameters
    ----------
    ula : UniformLinearArray
    props : TransducerProperty
    medium : Medium
    tx_delays : array_like, shape=(ula.num,).
        Transmit time delays.
    tx_apodization : array_like, shape=(ula.num,)
        Transmit apodization.
    scatterers : array_like, shape=(m, 2)
        Locations of the scatterers.
    reflectivity : array_like, shape=(m,)
        The reflectivity of the scatterers.
    pulse_t : array_like, shape=(n_samples,)
        Pulse emitted by the ULA, in time domain.
    sampling_rate : float
        Sampling rate.
    subelement_num : int or None, optional
        The number of sub-elements. Default is None, which automatically
        calculates the number of sub-elements.
    db_thresh : float, optional
        The frequency components whose amplitude is below db_thresh are ignored
        to improve performance.
    workers : int or any type, keyword-only, optional
        If `workers != 1`, then `concurrent.futures.ProcessPoolExecutor` is
        used and `workers` is passed as `max_workers` to
        `concurrent.futures.ProcessPoolExecutor`. The default is 1, that
        parallel computing is not used.
    show_pbar : bool, keyword-only, optional
        If True, show a progress bar. Default is False.

    Returns
    -------
    signal_rx_t : numpy.ndarray, shape=(ula.num, len(pulse_t))
        The simulated RF signals, where signal_rx_t[i, j] is the signal
        received by the i-th element of the ULA at time[j].

    """
    return get_p_scattered_t(
        ula, props, medium, tx_delays, tx_apodization,
        ula.pc, scatterers, reflectivity, pulse_t, sampling_rate,
        db_thresh=db_thresh, subelement_num=subelement_num,
        workers=workers, show_pbar=show_pbar,
    ).real


def rf2iq(signals_rx_t, fc, sampling_rate):
    """
    Return the I/Q components of the RF signals.

    Parameters
    ----------
    signals_rx_t : array_like
        RF signals. `signals_rx_t` can be 2D or 3D. The last dimension of
        `signals_rx_t` corresponds to fast-time.
    fc : float
        Center frequency of the pulse.
    sampling_rate : float
        Sampling rate.

    Returns
    -------
    iq_t : complex numpy.ndarray, shape=signals_rx_t.shape
        A complex array whose real/imaginary part contains the
        in-phase/quadrature components.

    """
    n_samples = signals_rx_t.shape[-1]
    duration = n_samples/sampling_rate
    time = np.linspace(0, duration, n_samples)
    iq = 2*signals_rx_t*np.exp(-1j*math.tau*fc*time)
    b, a = butter(5, Wn=min(2*fc/sampling_rate, 0.5))
    return filtfilt(b, a, iq, axis=signals_rx_t.ndim-1)


def time_gain_compensate(signal_t):
    """
    Return time-gain compensate RF or I/Q signals.

    Parameters
    ----------
    signals_t : array_like
        RF or I/Q signals. `signals_t` can be 2D or 3D. The last dimension of
        `signals_t` corresponds to fast-time.

    Returns
    -------
    result : numpy.ndarray, shape=signals_t.shape
        The compensated signals.

    Notes
    -----
    The signals are assumed to attenuate exponentially. The attenuation factor
    is obtained by least square fitting.

    """
    n_samples = signal_t.shape[-1]
    time = np.arange(n_samples)
    if np.isrealobj(signal_t):  # RF signals.
        amp = hilbert(signal_t, axis=signal_t.ndim-1)
    else:  # I/Q signals.
        amp = signal_t
    envelope = np.abs(amp).mean(axis=tuple(range(signal_t.ndim-1)))
    n1 = math.ceil(0.1*n_samples)
    n2 = math.ceil(0.9*n_samples)
    time_trunc = time[n1:n2]
    envelope_trunc = envelope[n1:n2]
    dbl_trunc = 20*np.log10(envelope_trunc/envelope_trunc.max())
    idx = dbl_trunc >= -40
    (slope, intercept), _, _, _ = np.linalg.lstsq(
        np.transpose([time_trunc, np.ones_like(time_trunc)])[idx],
        dbl_trunc[idx],
        rcond=None,
    )
    factor = 10**((slope*time+intercept)/20)
    return signal_t/factor


def _get_travel_time(ula, medium, tx_delays, points):
    d_rx = cdist(points, ula.pc)
    d_tx = (tx_delays*medium.c+d_rx).min(axis=1)
    return (d_tx[:, np.newaxis]+d_rx)/medium.c


def beamform(
    ula, medium, tx_delays, signals_t, sampling_rate, points,
    fnumber=None, fc=None,
):
    """
    Return the beamformed signal from the given RF or I/Q signals.

    Parameters
    ----------
    ula : UniformLinearArray
    medium : Medium
    tx_delays : array_like, shape=(ula.num,).
        Transmit time delays.
    signals_t : array_like, shape=(ula.num, n_samples)
        Signals of the ULA elements, in time domain. `signals_t` can be either
        RF signals (real signals) or I/Q signals (complex signals).
    sampling_rate : float
        Sampling rate.
    points : array_like, shape=(n, 2)
        Spatial points where the signals are beamformed.
    fnumber : None or float, optional
        The receive f-number of the transducer. The default is None, that is,
        the effect of receive aperture is ignored.
    fc : None or float, optional
        The center frequency of the pulse, default is None. `fc` is only
        required when `signals_t` is I/Q signals (complex signals). If
        `signals_t` is RF signals (real signals), `fc` should be None.

    Returns
    -------
    beamformed_signal : numpy.ndarray, shape=(len(points),)
        The beamformed signal, where beamformed_signal[i] corresponds to
        points[i]. The dtype of beamformed_signal is the same as that of
        signals_t.

    Notes
    -----
    This function is designed for fast beamforming when there are only a few
    frames of signals. It uses direct interpolation to calculate the delayed
    signals, which is faster than constructing a DAS matrix when the number of
    frames is small. For a large number of frames, it is recommended to
    precompute the DAS matrix with `get_dasmtx` and use `beamform_by_dasmtx`
    for maximum efficiency.

    """
    travel_times_T = _get_travel_time(ula, medium, tx_delays, points).T

    # Interpolation.
    signals_t = np.asarray(signals_t)
    n_samples = signals_t.shape[-1]
    time = np.arange(n_samples)/sampling_rate
    delayed = [
        np.interp(travel_times_T[i], time, signals_t[i])
        for i in range(ula.num)
    ]
    delayed = np.asarray(delayed)

    if fnumber is not None:
        aperture = (
            points[:, 1]
            >= 2*fnumber*np.abs(points[:, 0]-ula.xc[:, np.newaxis])
        )
        delayed[~aperture] = 0

    if np.iscomplexobj(signals_t):  # I/Q signals.
        if fc is None:
            raise ValueError('fc is required when signals_t is I/Q signals.')
        delayed *= np.exp(1j*math.tau*fc*travel_times_T)

    return delayed.sum(axis=0)


def get_dasmtx(
    ula, medium, tx_delays, n_samples, sampling_rate, points,
    fnumber=None, is_iq=False, fc=None,
):
    """
    Return the delay-and-sum (DAS) matrix.

    Parameters
    ----------
    ula : UniformLinearArray
    medium : Medium
    tx_delays : array_like, shape=(ula.num,).
        Transmit time delays.
    n_samples : int
        Number of samples in the signals
    sampling_rate : float
        Sampling rate.
    points : array_like, shape=(n, 2)
        Spatial points where the signals are beamformed.
    fnumber : None or float, optional
        The receive f-number of the transducer. The default is None, that is,
        the effect of receive aperture is ignored.
    is_iq : bool, optional
        Whether the signals are I/Q signals, default is False.
    fc : None or float, optional
        The center frequency of the pulse. fc is only required when the
        signals are I/Q signals. If is_iq == False, then fc is ignored.

    Returns
    -------
    dasmtx : scipy.sparse.csr_array, shape=(n, ula.num*n_samples)
        The sparse delay-and-sum (DAS) matrix. If is_iq == True, then dasmtx is
        a complex matrix.

    """
    travel_times = _get_travel_time(ula, medium, tx_delays, points)

    n_points = len(points)
    position = travel_times*sampling_rate
    relative_pos, idx = np.modf(position)
    idx = idx.astype(int)
    mask = (idx >= 0) & (idx < n_samples-1)

    if fnumber is not None:
        aperture = (
            points[:, 1, np.newaxis]
            >= 2*fnumber*np.abs(points[:, 0, np.newaxis]-ula.xc)
        )
        mask &= aperture

    row_idx = np.tile(np.arange(n_points, dtype=int)[:, np.newaxis], ula.num)
    col_idx = idx+n_samples*np.arange(ula.num, dtype=int)
    coeff1 = 1-relative_pos
    coeff2 = relative_pos

    if is_iq:  # I/Q signals.
        phase_factor = np.exp(1j*math.tau*fc*travel_times)
        coeff1 = coeff1*phase_factor
        coeff2 = coeff2*phase_factor

    row_idx = row_idx[mask]
    col_idx = col_idx[mask]
    coeff1 = coeff1[mask]
    coeff2 = coeff2[mask]

    row_idx = np.hstack([row_idx, row_idx])
    col_idx = np.hstack([col_idx, col_idx+1])
    data = np.hstack([coeff1, coeff2])

    return csr_array(
        (data, (row_idx, col_idx)), shape=(n_points, ula.num*n_samples),
    )


def beamform_by_dasmtx(dasmtx, signals_t):
    """
    Beamform the signals using the delay-and-sum (DAS) matrix.

    dasmtx and signals_t should be consistent in terms of whether it is I/Q
    signals.

    Parameters
    ----------
    dasmtx : scipy.sparse.csr_array
        The sparse delay-and-sum (DAS) matrix.
    signals_t : array_like, 2D or 3D
        Signals of the ULA elements, in time domain. If signals_t is 2D, the
        first and second dimensions correspond to element number and fast-time,
        respectively. If signals_t is 3D, the first, second and third
        dimensions correspond to slow-time, element number and fast-time,
        respectively.

    Returns
    -------
    beamformed_signal : numpy.ndarray
        The beamformed signal. If signals_t is 2D, then beamformed_signal is an
        1D array with beamformed_signal[i] corresponds to points[i]. If
        signals_t is 3D, then beamformed_signal is an 2D array with
        beamformed_signal[i, j] corresponds to frame[i] and points[j].

    """
    signals_t = np.asarray(signals_t)
    if signals_t.ndim == 2:
        return dasmtx@(signals_t.reshape((-1,)))
    return (dasmtx@signals_t.reshape((signals_t.shape[0], -1)).T).T


def log_compress(img, dynamic_range, uint8=False):
    """
    Return the log-compressed ultrasound image.

    Parameters
    ----------
    img : array_like
        The input image, which can be of any shape, either real or complex. The
        absolute value of the input image is log-compressed.
    dynamic_range : float
        The dynamic range in decibels.
    uint8 : bool, optional
        If True, the image is converted to 8-bit format, default is False.

    Returns
    -------
    image : numpy.ndarray, shape=img.shape
        The log-compressed image. If `uint8` is True, the dtype of the image is
        uint8, with values between 0 and 255. Otherwise, the dtype of the image
        is float, with values between -dynamic_range and 0.

    """
    img = np.abs(img)
    output = np.full(img.shape, -dynamic_range)
    img_max = img.max()
    if img_max > 0:
        mask = img > 0
        output = np.full(img.shape, -dynamic_range)
        output[mask] = np.clip(20*np.log10(img[mask]/img_max), -dynamic_range, 0)
    if uint8:
        return np.interp(output, [-dynamic_range, 0], [0, 255]).astype(np.uint8)
    return output


def gamma_compress(img, gamma, uint8=False):
    """
    Return the gamma-compressed image.

    Parameters
    ----------
    img : array_like
        The input image, which can be of any shape, either real or complex. The
        absolute value of the input image is gamma-compressed.
    gamma : float
        The gamma value.
    uint8 : bool, optional
        If True, the image is converted to 8-bit format, default is False.

    Returns
    -------
    image : numpy.ndarray, shape=img.shape
        The gamma-compressed image. If `uint8` is True, the dtype of the image
        is uint8, with values between 0 and 255. Otherwise, the dtype of the
        image is float, with values between 0 and 1.

    """
    img = np.abs(img)
    img_max = img.max()
    if img_max > 0:
        output = (img/img_max)**gamma
    else:
        output = np.zeros(img.shape)
    if uint8:
        return np.interp(output, [0, 1], [0, 255]).astype(np.uint8)
    return output


def genscat(x, z, v, meandist, rng):
    """
    Generate scatterers and their reflectivity based on a reflectivity
    grayscale image.

    This function is a simplified version of "genscat" from the MATLAB
    Ultrasound Toolbox (MUST) by Damien Garcia.

    Parameters
    ----------
    x : numpy.ndarray, shape=(nx,)
        The x-coordinates of the reflectivity grayscale image.
    z : numpy.ndarray, shape=(nz,)
        The z-coordinates of the reflectivity grayscale image.
    v : numpy.ndarray, shape=(nz, nx)
        The reflectivity grayscale image.
    meandist : float
        The mean distance between scatterers.
    rng : numpy.random.Generator
        The random number generator.

    Returns
    -------
    scatterers : numpy.ndarray, shape=(n, 2)
        The locations of the scatterers.
    reflectivity : numpy.ndarray, shape=(n,)
        The reflectivity of the scatterers.

    """
    xmin, xmax = x.min(), x.max()
    zmin, zmax = z.min(), z.max()
    xz_inc = meandist/math.sqrt(2/5)
    x_mesh, z_mesh = np.meshgrid(
        np.arange(xmin, xmax, xz_inc),
        np.arange(zmin, zmax, xz_inc),
        indexing='xy',
    )
    scatterers = np.transpose([x_mesh.reshape((-1,)), z_mesh.reshape((-1,))])
    scatterers += rng.uniform(-0.5, 0.5, scatterers.shape)*xz_inc
    scatterers = scatterers[
        (scatterers[:, 0] > xmin) &
        (scatterers[:, 0] < xmax) &
        (scatterers[:, 1] > zmin) &
        (scatterers[:, 1] < zmax)
    ]
    reflectivity = interpn((x, z), v, scatterers)
    # Log compression.
    reflectivity = 10**(40/20*(reflectivity-1))
    # Rayleigh distribution.
    reflectivity = reflectivity*rng.rayleigh(size=reflectivity.shape)
    return (scatterers, reflectivity)
