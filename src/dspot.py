import numpy as np

from pot import pot
from utils.grimshaw import grimshaw


def dspot(data:np.array, num_init:int, depth:int, risk:float):
    ''' Streaming Peak over Threshold with Drift

    Reference:
    Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory." 
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge 
    Discovery and Data Mining. 2017.

    Args:
        data: data to process
        num_init: number of data point selected to init threshold
        depth: number of data point selected to detect drift
        risk: detection level

    Returns: 
        logs: 't' threshold with dataset length; 'a' anomaly datapoint index
    '''
    logs = {'t': [], 'a': []}

    base_data = data[:depth]
    init_data = data[depth:depth + num_init]
    rest_data = data[depth + num_init:]

    for i in range(num_init):
        temp = init_data[i]
        init_data[i] -= base_data.mean()
        np.delete(base_data, 0)
        np.append(base_data, temp)

    z, t = pot(init_data)
    k = num_init
    peaks = init_data[init_data > t] - t
    logs['t'] = [z] * (depth + num_init)

    for index, x in enumerate(rest_data):
        temp = x
        x -= base_data.mean()
        if x > z:
            logs['a'].append(index + num_init + depth)
        elif x > t:
            peaks = np.append(peaks, x - t)
            gamma, sigma = grimshaw(peaks=peaks, threshold=t)
            k = k + 1
            r = k * risk / peaks.size
            z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
            np.delete(base_data, 0)
            np.append(base_data, temp)
        else:
            k = k + 1
            np.delete(base_data, 0)
            np.append(base_data, temp)

        logs['t'].append(z)
    
    return logs