import numpy as np 

from math import log
from peak_over_threshold.utils.grimshaw import Grimshaw
from peak_over_threshold.utils.cal_threshold import CalcThreshold


def PeakOverThreshold(
        data: np.array, 
        num_candidates: int,
        risk: float = 1e-4, 
        init_level: float = 0.98, 
        epsilon: float = 1e-8
    ):
    """ Peak-over-Threshold Alogrithm

    Args:
        data <np.array> [sequence_length,]: 1-D numpy array data to process
        num_candidates <int>: the maximum number of nodes we choose as candidates
        risk <float>: detection level, i.e. q in the original paper
        init_level <float>: probability associated with the initial threshold
        epsilon <float>: numerical parameter to perform
    
    Returns:
        z <float>: threshold searched by pot
        t <float>: init threshold 

    References: 
        Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory." 
        Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge 
        Discovery and Data Mining. 2017.
    """
    t = SetInitialThreshold(data, init_level)

    # Find Peaks
    y = data[data > t] - t

    # Grimshaw
    gamma, sigma = Grimshaw(
        peaks=y, 
        threshold=t, 
        num_candidates=num_candidates, 
        epsilon=epsilon
    )

    # Calculate the Threshold
    z = CalcThreshold(
        q = risk,
        gamma = gamma,
        sigma = sigma,
        n = y.size,
        N = data.size,
        t = t,
    )

    return z, t
    

def SetInitialThreshold(x: np.array, init_level: float = 0.98):
    """
    Follow the paper section 4.3.3 to set the initial threshold with:
    "In  practice we set t to a high empirical quantile (98%)."
    """
    t = np.sort(x)[int(init_level * len(x))]
    return t
