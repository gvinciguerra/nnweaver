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
