import numpy as np


class Activation(object):
    def __call__(self, x):
        raise NotImplementedError

    def gradient(self, x):
        raise NotImplementedError


class Linear(Activation):
    def __call__(self, x):
        """ Compute the linear activation function (identity) for each
        element of :param:`x`.

        :param x: an array.
        :return: :param:`x`.
        """
        return x

    def gradient(self, x):
        """ Compute the gradient of the linear activation function
        (constant to 1) for each element of :param:`x`.

        :param x: an array.
        :return: an array of 1s of the same shape of :param:`x`.
        """
        return np.ones(x.shape)


class Rectifier(Activation):
    def __call__(self, x):
        """ Compute the rectifier activation function for each element
        of :param:`x`.

        :param x: an array.
        :return: the activated values of :param:`x`.
        """
        return np.multiply(x, x > 0)

    def gradient(self, x):
        """ Compute the gradient of the rectifier activation function for
        each element of :param:`x`.

        :param x: an array.
        :return: the gradients on the values of :param:`x`.
        """
        return np.multiply(1, x > 0)


class Sigmoid(Activation):
    def __call__(self, x):
        """ Compute the sigmoid activation function for each element of :param:`x`.

        :param x: an array.
        :return: the activated values of :param:`x`.
        """
        return 1.0 / (1.0 + np.exp(-x))

    def gradient(self, x):
        """ Compute the gradient of the sigmoid activation function for
        each element of :param:`x`.

        :param x: an array.
        :return: the gradients on the values of :param:`x`.
        """
        y = self(x)
        return np.multiply(y, 1 - y)
