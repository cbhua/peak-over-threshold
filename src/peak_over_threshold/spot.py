import numpy as np 

from peak_over_threshold.pot import PeakOverThreshold
from peak_over_threshold.utils.grimshaw import Grimshaw
from peak_over_threshold.utils.cal_threshold import CalcThreshold


def StreamingPeakOverThreshold(
        data: np.array, 
        num_init: int, 
        num_candidates: int,
        risk: float = 1e-4, 
        init_level: float = 0.98, 
        epsilon: float = 1e-8,
    ):
    """ Streaming Peak over Threshold Algorithm

    Args:
        data <np.array> [sequence_length,]: 1-D numpy array data to process
        num_init <int>: number of data point selected to init threshold
        num_candidates <int>: the maximum number of nodes we choose as candidates
        risk <float>: detection level
        init_level <float>: probability associated with the initial threshold
        epsilon <float>: numerical parameter to perform

    Returns:
        z_spot <np.array>: threshold searched by spot

    Reference: 
        Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory." 
        Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge 
        Discovery and Data Mining. 2017.
    """
    init_data = data[:num_init]
    rest_data = data[num_init:]

    z, t = PeakOverThreshold(data, num_candidates, risk, init_level, epsilon)
    
    # Find initial peaks
    y = init_data[init_data > t] - t
    
    # Init the threshold for SPOT
    z_spot = [z] * num_init

    for idx, x in enumerate(rest_data):
        if x > z: # anormaly case
            pass
        elif x > t: # real peak case
            # Add new peak
            y = np.append(y, x - t)
            gamma, sigma = Grimshaw(
                peaks=y, 
                threshold=t,
                num_candidates=num_candidates,
                epsilon=epsilon,
            )
            z = CalcThreshold(
                q = risk,
                gamma = gamma,
                sigma = sigma,
                n = y.size,
                N = num_init + idx,
                t = t,
            )
        else: # normal case
            pass

        z_spot.append(z)
    
    return z_spot