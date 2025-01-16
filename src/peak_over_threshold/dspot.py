import numpy as np

from peak_over_threshold.pot import PeakOverThreshold
from peak_over_threshold.utils.grimshaw import Grimshaw
from peak_over_threshold.utils.cal_threshold import CalcThreshold


def DriftStreamingPeakOverThreshold(
        data: np.array, 
        num_init: int, 
        depth: int,
        num_candidates: int,
        risk: float = 1e-4, 
        init_level: float = 0.98, 
        epsilon: float = 1e-8,
    ):
    """ Streaming Peak over Threshold with Drift

    Args:
        data <np.array> [sequence_length,]: 1-D numpy array data to process
        num_init <int>: number of data point selected to init threshold
        depth <int>: number of data point selected to detect drift
        num_candidates <int>: the maximum number of nodes we choose as candidates
        risk <float>: detection level
        init_level <float>: probability associated with the initial threshold
        epsilon <float>: numerical parameter to perform

    Returns:
        z_dspot <np.array>: threshold searched by spot

    Reference:
        Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory." 
        Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge 
        Discovery and Data Mining. 2017.
    """
    # Last depth normal values
    w = data[:depth]

    # Local model with depth
    m = w.mean()

    x_ = []
    for i in range(depth, depth + num_init):
        x_.append(data[i] - m)
        m = data[i - depth + 1:i].mean()
    x_ = np.array(x_)

    z, t = PeakOverThreshold(
        data=x_,
        num_candidates=num_candidates,
        risk=risk,
        init_level=init_level,
        epsilon=epsilon
    )

    # Find the peaks
    y = x_[x_ > t] - t

    # Record the threshold
    z_list = [z] * (depth + num_init)

    for i in range(depth + num_init, data.size):
        x_ = np.append(x_, data[i] - m)
        if x_[-1] > z: # anormaly case
            z_list.append(z)
            continue
        elif x_[-1] > t: # real peak case
            y = np.append(y, x_[-1] - t)
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
                N = len(x_),
                t = t,
            ) 
        else: # normal case
            pass

        # Add the current normal value
        w = w[1:]
        w = np.append(w, data[i])

        # Update the local model
        m = w.mean()

        # Record the threshold
        z_list.append(z)

    return np.array(z_list)
