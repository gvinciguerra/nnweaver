""" The :py:mod:`.losses` module provides a set of loss/objective
functions whose output should be minimized during the training phase. """

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    """ Abstract base class for classes that provide methods to compute the
    cost function over a single example or a batch of examples.
    """

    @classmethod
    @abstractmethod
    def loss(cls, y, y_target):
        pass

    @classmethod
    @abstractmethod
    def gradient(cls, y, y_target):
        pass

    @classmethod
    def batch_mean(cls, y, y_target):
        """ Compute the average loss over a batch of examples.

        :param y: the bi-dimensional matrix whose rows are the examples and
            whose columns are the predicted features.
        :param y_target: the target bi-dimensional matrix. Must have the same
            number of rows of ``y``,
        :return: the average loss.
        """
        tot = 0
        for i, o in zip(y, y_target):
            tot += cls.loss(i.reshape(-1, 1), o.reshape(-1, 1))
        return tot / len(y)


class MSE(Loss):
    @classmethod
    def loss(cls, y, y_target):
        """ Compute the Mean Squared Error (MSE) of the predicted example.

        :param y: the predicted output (a column vector).
        :param y_target: the target output (a column vector).
        :return: the MSE.
        """
        return 0.5 * np.sum(np.square(y - y_target))

    @classmethod
    def gradient(cls, y, y_target):
        """ Compute the gradient of Mean Squared Error (MSE) of the predicted
        example.

        :param y: the predicted output (a column vector).
        :param y_target: the target output (a column vector).
        :return: the gradient of the MSE (a column vector).
        """
        return y - y_target


class MEE(Loss):
    @classmethod
    def euclidean_distance(cls, y, y_target):
        """ Compute the euclidean distance between two vectors.

        :param y: the predicted output (a column vector).
        :param y_target: the target output (a column vector).
        :return: the euclidean distance.
        """
        return np.linalg.norm(y - y_target)

    @classmethod
    def loss(cls, y, y_target):
        """ Compute the Mean Euclidean Error (MEE) of the predicted example.

        :param y: the predicted output (a column vector).
        :param y_target: the target output (a column vector).
        :return: the MEE.
        """
        return MEE.euclidean_distance(y, y_target)

    @classmethod
    def gradient(cls, y, y_target):
        """ Compute the gradient of Mean Euclidean Error (MEE) of the predicted
        example.

        :param y: the predicted output (a column vector).
        :param y_target: the target output (a column vector).
        :return: the gradient of the MEE (a column vector).
        """
        return (y - y_target) / MEE.euclidean_distance(y, y_target)
