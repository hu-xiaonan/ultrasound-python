import os
import sys
from pathlib import Path
import math
import numpy as np
import timeit

os.chdir(Path(__file__).parent.resolve())
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from sonic import _get_integral_over_h  # noqa: E402


def fast_version(Rf, h, k, r):
    """
    Compute the integral using the Gaussian superposition model described in 
    Eq. (11) of the [SIMUS](https://doi.org/10.1016/j.cmpb.2022.106726) paper. 
    This method is claimed to be faster than the accurate approach that relies 
    on the Gaussian error function.

    """
    A = np.asarray(
        [0.187+0.275j, 0.288-1.954j, 0.187-0.275j, 0.288+1.954j]
    )
    B = np.asarray(
        [4.558-25.59j, 8.598-7.924j, 4.558+25.59j, 8.598+7.924j]
    )
    A = A[:, np.newaxis, np.newaxis, np.newaxis]
    B = B[:, np.newaxis, np.newaxis, np.newaxis]
    alpha = B/h**2+1j/2*k*(1/Rf-1/r)
    return np.sum(A*np.sqrt(math.pi/alpha), axis=0)


if __name__ == '__main__':
    Rf = 60e-3
    h = 14e-3
    k = 1000
    r = np.random.uniform(0, 10e-2, (1000, 1000))

    n1, t1 = timeit.Timer(stmt='_get_integral_over_h(Rf, h, k, r)', globals=globals()).autorange()
    n2, t2 = timeit.Timer(stmt='fast_version(Rf, h, k, r)', globals=globals()).autorange()
    print(f'Calculating accurate results using erf: {t1/n1:.3g} s')
    print(f'Calculating approx. results using Gaussian superposition: {t2/n2:.3g} s')
