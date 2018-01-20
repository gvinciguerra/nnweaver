import numpy as np


def one_hot_encoding(x):
    result = np.zeros((x.size, np.max(x) + 1))
    result[np.arange(x.size), x] = 1
    return result


def accuracy(y, y_target):
    return (np.abs(y - y_target) < 0.5).mean()
