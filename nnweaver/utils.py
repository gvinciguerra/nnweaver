import numpy as np


def one_hot_encoding(x):
    """ Compute the One-Hot-Encoding (OHE) of a list of elements.

    :param x: the elements to be encoded.
    :return: a matrix with rows equal to the number of the
        elements in :param:`x` and max(:param:`x`) columns.
    """
    result = np.zeros((x.size, np.max(x) + 1))
    result[np.arange(x.size), x] = 1
    return result


def accuracy(y, y_target):
    """ Compute the (binary) accuracy of a given prediction.

    :param y: the predicted output.
    :param y_target: the target (binary) output.
    :return: the accuracy of the predicted output, rounded.
    """
    y_target = y_target.reshape(y.shape)
    return (np.abs(y - y_target) < 0.5).mean()
