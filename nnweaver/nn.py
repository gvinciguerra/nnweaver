""" The :py:mod:`.nn` module contains the :py:class:`.NN` (neural network) and
the :py:class:`.Layer` classes.

Formally, a neural network models a function
:math:`f(\\boldsymbol x)=f^{(L)}(f^{(L-1)}(\\dots f^{(1)}(\\boldsymbol
x)\\dots ))` , where :math:`\\boldsymbol x` is the input vector, and the
:math:`f^{(i)}` s are the layers of the network.

A layer specifies a function
:math:`f(\\boldsymbol x)=g(\\boldsymbol{Wx} + \\boldsymbol b)` ,
where :math:`g` is called activation function, :math:`\\boldsymbol W` provides
the weights of a linear transformation, and :math:`\\boldsymbol b` is the bias
vector.

A neural network can be trained with an optimizer and then used to predict the
target values of (previously unseen) examples. See the :py:mod:`.optimizers`
module for a list of available training algorithms and the
:py:mod:`.activations` module for a list of available activation functions.
"""

import copy

import numpy as np

from .activations import Linear


class NN(object):
    def __init__(self, input_dim):
        """ Create a multi-layer perceptron (MLP, also called feedforward
        neural network) with the given number of inputs.

        :param input_dim: number of dimensions of the input.
        """
        self.input_dim = input_dim
        self.layers = []

    def add_layer(self, layer):
        """ Add a layer to the neural network.

        :param layer: the :py:class:`.Layer` to add.
        """
        if len(self.layers) == 0:
            layer.build_weights(self.input_dim)
        else:
            preceding_units = self.layers[-1].units
            layer.build_weights(preceding_units)

        self.layers.append(layer)

    def predict(self, x):
        """ Feed a single input to the neural network.

        :param x: the input. Its size must match the ``input_dim`` of the neural
            network.
        :return: the output of the neural network (a column vector).
        """
        xi = x
        for l in self.layers:
            xi = l(xi)

        return xi

    def predict_batch(self, x_batch):
        """ Feed multiple inputs to the neural network.

        :param x_batch: a list of examples.
        :return: an array (with the same size of x_batch) with the outputs of
            the network.
        """
        return np.array([self.predict(i.reshape(-1, 1)) for i in x_batch]) \
            .reshape(-1, self.layers[-1].units)

    def reset(self):
        """ Flash the neural network with a neuralyzer. """
        for l in self.layers:
            l.reset_weights()

    def clone(self):
        """ Clone the neural network.

        :return: the copy of the neural network.
        """
        return copy.deepcopy(self)


class Layer(object):
    def __init__(self, units, activation=Linear):
        """ Create a new layer with a given number of units and activation
        function.

        :param units: the number of units in the layer.
        :param activation: the activation function.
        """
        assert units > 0
        self.units = units
        self.bias = None
        self.weights = None
        self.activation = activation

    def build_weights(self, preceding_units, low=-0.05, high=0.05):
        """ Initialize the weights and the bias vector of the layer with random
        values.

        :param preceding_units: number of units in the previous level.
        :param low: lower boundary of the random values.
        :param high: upper boundary of the random values.
        """
        if self.weights is not None:
            raise Exception("This layer was already added to a NN")

        self.bias = np.random.uniform(low, high, (self.units, 1))
        self.weights = np.random.uniform(low, high, (self.units, preceding_units))

    def input_sum(self, x):
        """ Compute the input to the layer, without applying the activation
        function.

        :param x: input to the layer (a column vector).
        :return: weighted sum of the input with the bias (a column vector).
        """
        return self.bias + self.weights.dot(x).reshape(self.bias.shape)

    def reset_weights(self, low=-0.05, high=0.05):
        """ Re-initialize the weights and the bias vector of the layer with
        random values.

        :param low: lower boundary of the random values.
        :param high: upper boundary of the random values.
        """
        self.bias = np.random.uniform(low, high, self.bias.shape)
        self.weights = np.random.uniform(low, high, self.weights.shape)

    def __call__(self, x):
        """ Compute the output of the layer.

        :param x: input to the layer (a column vector).
        :return: activated output (a column vector).
        """
        input_sum = self.input_sum(x)
        return self.activation.apply(input_sum)
