""" The :py:mod:`.regularizers` provides a set of weight regularization
classes to be used during the training of a neural network. """

from abc import abstractmethod, ABC

import numpy as np


class Regularizer(ABC):
    """ Abstract base class for classes that implements functions on the
    :py:class:`.NN`'s parameters to control the complexity of the model during
    the training.
    """

    @abstractmethod
    def __call__(self, nn):
        pass

    @abstractmethod
    def gradient(self, layer):
        pass


class L1L2Regularizer(Regularizer):
    def __init__(self, l1=0, l2=0):
        """ Initialize a regularizer on weights' norms.

        :param l1: the factor of the L1 weight decay term.
        :param l2: the factor of the L2 weight decay (aka ridge regression or
            Tikhonov regularization) term.
        """
        self.l1 = l1
        self.l2 = l2

    def __call__(self, nn):
        """ Apply both L1 and L2 norms on the weights of every layer of the
        given neural network.

        :param nn: a neural network.
        :return: the loss penalty.
        """
        loss_penalty = 0
        for layer in nn.layers:
            concat = np.concatenate((layer.weights, layer.bias), axis=1)
            if self.l1 != 0:
                loss_penalty = self.l1 * np.linalg.norm(concat, ord=1, axis=1).mean()
            if self.l2 != 0:
                loss_penalty += 0.5 * self.l2 * np.linalg.norm(concat, ord=2, axis=1).mean()
        return loss_penalty

    def gradient(self, layer):
        """ Compute the gradient of the loss penalty for a single layer of a
        neural network.

        :param layer: a layer.
        :return: the weight and the bias gradient penalty.
        """
        weight_penalty = 0
        bias_penalty = 0
        if self.l1 != 0:
            weight_penalty = self.l1 * np.sign(layer.weights)
            bias_penalty = self.l1 * np.sign(layer.bias)
        if self.l2 != 0:
            weight_penalty += self.l2 * layer.weights
            bias_penalty += self.l2 * layer.bias
        return weight_penalty, bias_penalty
