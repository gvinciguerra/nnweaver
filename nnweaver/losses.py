import numpy as np


class Loss(object):
    def __call__(self, Y, Y_target):
        raise NotImplementedError

    def gradient(self, Y, Y_target):
        raise NotImplementedError


class MSE(Loss):
    def __call__(self, Y, Y_target):
        raise np.square(Y - Y_target).mean()

    def gradient(self, Y, Y_target):
        raise NotImplementedError
