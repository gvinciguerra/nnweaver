""" The :py:mod:`nnweaver.losses` module provides a set of loss/objective
functions whose output should be minimized during the training phase. """

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @classmethod
    @abstractmethod
    def __call__(cls, y, y_target):
        pass

    @classmethod
    @abstractmethod
    def gradient(cls, y, y_target):
        pass


class MSE(Loss):
    @classmethod
    def __call__(cls, y, y_target):
        """ Compute the Mean Squared Error (MSE) of the predicted data.

        :param y: the predicted output.
        :param y_target: the target output.
        :return: the MSE.
        """
        return 0.5 * np.square(y - y_target).mean()

    @classmethod
    def gradient(cls, y, y_target):
        """ Compute the gradient of Mean Squared Error (MSE) of the predicted
        data.

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
        """ Compute the gradient of Mean Euclidean Error (MEE) of the predicted
        data.

        :param y: the predicted output.
        :param y_target: the target output.
        :return: the gradient of the MEE.
        """
        return (y - y_target) / MEE.euclidean_distance(y, y_target)
