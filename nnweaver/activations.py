""" The :py:mod:`.activations` module provides a set of activation
functions to be applied to a layer of a neural network. """

from abc import ABC, abstractmethod

import numpy as np


class Activation(ABC):
    """ Abstract base class for classes that implement an activation function.
    """

    @classmethod
    @abstractmethod
    def apply(cls, x):
        pass

    @classmethod
    @abstractmethod
    def gradient(cls, x):
        pass


class Linear(Activation):
    @classmethod
    def apply(cls, x):
        """ Compute the linear activation function (identity) on each element of
        the input array.

        :param x: an array.
        :return: the input array.
        """
        return x

    @classmethod
    def gradient(cls, x):
        """ Compute the gradient of the linear activation function (constant to
        1) on each element of the input array.

        :param x: an array.
        :return: an array of 1s of the same shape of the input array.
        """
        return np.ones(x.shape)


class Rectifier(Activation):
    @classmethod
    def apply(cls, x):
        """ Compute the rectifier activation function on each element of the
        input array.

        :param x: an array.
        :return: the activated values of the input array.
        """
        return np.multiply(x, x > 0)

    @classmethod
    def gradient(cls, x):
        """ Compute the gradient of the rectifier activation function on each
        element of the input array.

        :param x: an array.
        :return: the gradients on the values of the input array.
        """
        return np.multiply(1, x > 0)


class Sigmoid(Activation):
    @classmethod
    def apply(cls, x):
        """ Compute the sigmoid activation function on each element of the
        input array.

        :param x: an array.
        :return: the activated values of the input array.
        """
        return 1.0 / (1.0 + np.exp(-x))

    @classmethod
    def gradient(cls, x):
        """ Compute the gradient of the sigmoid activation function on each
        element of the input array.

        :param x: an array.
        :return: the gradients on the values of the input array.
        """
        y = Sigmoid.apply(x)
        return np.multiply(y, 1 - y)


class TanH(Activation):
    @classmethod
    def apply(cls, x):
        """ Compute the TanH activation function on each element of the input
        array.

        :param x: an array.
        :return: the activated values of the input array.
        """
        return np.tanh(x)

    @classmethod
    def gradient(cls, x):
        """ Compute the gradient of the TanH activation function on each element
        of the input array.

        :param x: an array.
        :return: the gradients on the values of the input array.
        """
        return 1 - TanH.apply(x) ** 2
