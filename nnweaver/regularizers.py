from abc import abstractmethod

import numpy as np


class Regularizer(object):
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
        loss_penalty = 0
        for layer in nn.layers:
            concat = np.concatenate((layer.weights, layer.bias), axis=1)
            if self.l1 != 0:
                loss_penalty = self.l1 * np.linalg.norm(concat, ord=1, axis=1).mean()
            if self.l2 != 0:
                loss_penalty += 0.5 * self.l2 * np.linalg.norm(concat, ord=2, axis=1).mean()
        return loss_penalty

    def gradient(self, layer):
        weight_penalty = 0
        bias_penalty = 0
        if self.l1 != 0:
            weight_penalty = self.l1 * np.sign(layer.weights)
            bias_penalty = self.l1 * np.sign(layer.bias)
        if self.l2 != 0:
            weight_penalty += self.l2 * layer.weights
            bias_penalty += self.l2 * layer.bias
        return weight_penalty, bias_penalty
