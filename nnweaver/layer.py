from numpy.random import uniform

from .activations import Linear


class Layer(object):
    def __init__(self, units, activation=Linear()):
        assert units > 0
        self.units = units
        self.weights = None
        self.activation = activation

    def build_weights(self, preceding_units, low=-0.05, high=0.05):
        if self.weights is not None:
            raise Exception("This layer was already added to a NN")

        self.weights = uniform(low, high, (preceding_units, self.units))
        self.bias = uniform(low, high, self.units)

    def __call__(self, x):
        out = self.bias + x.dot(self.weights)
        return self.activation(out)
