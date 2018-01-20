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
        return np.square(y - y_target).mean()

    @classmethod
    def gradient(cls, y, y_target):
        return y - y_target


class MEE(Loss):
    @classmethod
    def euclidean_distance(cls, y, y_target):
        return np.sqrt(np.sum(np.square(y - y_target), axis=1))

    @classmethod
    def __call__(cls, y, y_target):
        return MEE.euclidean_distance(y, y_target).mean()

    @classmethod
    def gradient(cls, y, y_target):
        return (y - y_target) / MEE.euclidean_distance(y, y_target)
