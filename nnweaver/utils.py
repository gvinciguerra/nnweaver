""" The :py:mod:`nnweaver.utils` module provides a set of utility functions. """

import numpy as np


def one_hot_encoding(x):
    """ Compute the One-Hot-Encoding (OHE) of a list of elements.

    :param x: the elements to be encoded.
    :return: a matrix with rows equal to the number of the
        elements in x and max(x) columns.
    """
    result = np.zeros((x.size, np.max(x) + 1))
    result[np.arange(x.size), x] = 1
    return result


def accuracy(y, y_target):
    """ Compute the (binary) accuracy of a given prediction.

    If the dimension of the input arrays is greater than 1 (e.g. `y_target` is
    encoded with :py:func:`.one_hot_encoding`), the multi-class accuracy is
    computed.

    :param y: the predicted output.
    :param y_target: the target (binary) output.
    :return: the accuracy of the predicted output, rounded.
    """
    y_target = y_target.reshape(y.shape)

    if y_target.ndim > 1 and y_target.shape[1] > 1:  # multi-class accuracy
        return (np.argmax(y, -1) == np.argmax(y_target, -1)).mean()

    return (np.abs(y - y_target) < 0.5).mean()
