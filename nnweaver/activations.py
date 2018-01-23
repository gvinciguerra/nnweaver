""" The :py:mod`nnweaver.activations` module provides a set of activation
functions to be applied to a layer of a neural network. """

import numpy as np


class Activation(object):
    def __call__(self, x):
        raise NotImplementedError

    def gradient(self, x):
        raise NotImplementedError


class Linear(Activation):
    def __call__(self, x):
        """ Compute the linear activation function (identity) on each element of
        the input array.

        :param x: an array.
        :return: the input array.
        """
        return x

    def gradient(self, x):
        """ Compute the gradient of the linear activation function (constant to
        1) on each element of the input array.

        :param x: an array.
        :return: an array of 1s of the same shape of the input array.
        """
        return np.ones(x.shape)


class Rectifier(Activation):
    def __call__(self, x):
        """ Compute the rectifier activation function on each element of the
        input array.

        :param x: an array.
        :return: the activated values of the input array.
        """
        return np.multiply(x, x > 0)

    def gradient(self, x):
        """ Compute the gradient of the rectifier activation function on each
        element of the input array.

        :param x: an array.
        :return: the gradients on the values of the input array.
        """
        return np.multiply(1, x > 0)


class Sigmoid(Activation):
    def __call__(self, x):
        """ Compute the sigmoid activation function on each element of the
        input array.

        :param x: an array.
        :return: the activated values of the input array.
        """
        return 1.0 / (1.0 + np.exp(-x))

    def gradient(self, x):
        """ Compute the gradient of the sigmoid activation function on each
        element of the input array.

        :param x: an array.
        :return: the gradients on the values of the input array.
        """
        y = self(x)
        return np.multiply(y, 1 - y)


class TanH(Activation):
    def __call__(self, x):
        """ Compute the TanH activation function on each element of the input
        array.

        :param x: an array.
        :return: the activated values of the input array.
        """
        return 2*Sigmoid()(2*x) - 1

    def gradient(self, x):
        """ Compute the gradient of the TanH activation function on each element
        of the input array.

        :param x: an array.
        :return: the gradients on the values of the input array.
        """
        return 1 - TanH()(x) ** 2
