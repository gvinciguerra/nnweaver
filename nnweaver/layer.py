import numpy as np

from .activations import Linear


class Layer(object):
    def __init__(self, units, activation=Linear()):
        assert units > 0
        self.units = units
        self.bias = None
        self.weights = None
        self.activation = activation

    def build_weights(self, preceding_units, low=-0.05, high=0.05):
        """ Initialize the weights and the biases.

        :param preceding_units: number of units in the previous level.
        :param low: lower boundary of the weights.
        :param high: upper boundary of the weights.
        """
        if self.weights is not None:
            raise Exception("This layer was already added to a NN")

        self.bias = np.random.uniform(low, high, (self.units, 1))
        self.weights = np.random.uniform(low, high, (self.units, preceding_units))

    def input_sum(self, x):
        """ Compute the input to the layer, without applying the activation
        function.

        :param x: input to the layer.
        :return: weighted sum of the input with the bias.
        """
        return self.bias + self.weights.dot(x).reshape(self.bias.shape)

    def __call__(self, x):
        """ Compute the output of the layer.

        :param x: input to the layer.
        :return: activated output.
        """
        input_sum = self.input_sum(x)
        return self.activation(input_sum)
