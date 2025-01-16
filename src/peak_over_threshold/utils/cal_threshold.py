import numpy as np

from warnings import warn


def CalcThreshold(
    q,
    gamma,
    sigma,
    n,
    N,
    t,
):
    """
    Args:
        - q: risk
        - gamma: estimate
        - sigma: estimate
        - n: the peak size
        - N: the whole data size
        - t: init threshold
    
    Return:
        - z: threshold
    """
    assert n != 0

    if gamma != 0:
        z = t + (sigma / gamma) * (pow(q * N / n, -gamma) - 1)
    else:
        warn("gamma is zero, the threshold is calculated by the original formula")
        z = t - sigma * np.log(q * N / n)

    return z
