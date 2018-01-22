import numpy as np


class NN(object):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.layers = []

    def add_layer(self, layer):
        """ Add a layer to the neural network.

        :param layer: the layer to add.
        """
        if len(self.layers) == 0:
            layer.build_weights(self.input_dim)
        else:
            preceding_units = self.layers[-1].units
            layer.build_weights(preceding_units)

        self.layers.append(layer)

    def predict(self, x):
        """ Feed a single input to the neural network.

        :param x: the dimension must match the input layer of the neural network.
        :return: the output of the network.
        """
        xi = x
        for l in self.layers:
            xi = l(xi)

        return xi

    def predict_batch(self, x_batch):
        """ Feed multiple inputs to the neural network.

        :param x_batch: an array of inputs.
        :return: an array (with the same size of x_batch) with the outputs of the network.
        """
        return np.array([self.predict(i.reshape(-1, 1)) for i in x_batch]).reshape(-1, 1)

    def reset(self):
        """ Flash the neural network with a neuralyzer. """
        for l in self.layers:
            l.reset_weights()
