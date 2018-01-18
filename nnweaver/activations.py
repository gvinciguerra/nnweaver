import numpy as np


class Activation(object):
    def __call__(self, x):
        raise NotImplementedError

    def gradient(self, x):
        raise NotImplementedError


class Linear(Activation):
    def __call__(self, x):
        return x

    def gradient(self, x):
        return 1


class Rectifier(Activation):
    def __call__(self, x):
        return np.multiply(x, x > 0)

    def gradient(self, x):
        return np.multiply(1, x > 0)


class Sigmoid(Activation):
    def __call__(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def gradient(self, x):
        y = self(x)
        return np.multiply(y, 1 - y)
