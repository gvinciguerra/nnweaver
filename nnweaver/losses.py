import numpy as np


class Loss(object):
    @classmethod
    def __call__(cls, y, y_target):
        raise NotImplementedError

    @classmethod
    def gradient(cls, y, y_target):
        raise NotImplementedError


class MSE(Loss):
    @classmethod
    def __call__(cls, y, y_target):
        """ Compute the Mean Squared Error (MSE) of the predicted data.

        :param y: the predicted output.
        :param y_target: the target output.
        :return: the MSE.
        """
        return np.square(y - y_target).mean()

    @classmethod
    def gradient(cls, y, y_target):
        """ Compute the gradient of Mean Squared Error (MSE) of the predicted data.

        :param y: the predicted output.
        :param y_target: the target output.
        :return: the gradient of the MSE.
        """
        return y - y_target


class MEE(Loss):
    @classmethod
    def euclidean_distance(cls, y, y_target):
        """ Compute the pairwise euclidean distances between the
        rows of two matrix of the same shape.

        :param y: the predicted output.
        :param y_target: the target output.
        :return: the euclidean distances.
        """
        return np.sqrt(np.sum(np.square(y - y_target), axis=1))

    @classmethod
    def __call__(cls, y, y_target):
        """ Compute the Mean Euclidean Error (MEE) of the predicted data.

        :param y: the predicted output.
        :param y_target: the target output.
        :return: the MEE.
        """
        return MEE.euclidean_distance(y, y_target).mean()

    @classmethod
    def gradient(cls, y, y_target):
        """ Compute the gradient of Mean Euclidean Error (MEE) of the predicted data.

        :param y: the predicted output.
        :param y_target: the target output.
        :return: the gradient of the MEE.
        """
        return (y - y_target) / MEE.euclidean_distance(y, y_target)
