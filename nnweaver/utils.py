import numpy as np


def one_hot_encoding(x):
    result = np.zeros((x.size, np.max(x) + 1))
    result[np.arange(x.size), x] = 1
    return result
