""" The :py:mod:`nnweaver.optimizers` module provides a set of optimization
algorithms that can be used to train neural networks. """

import itertools
from abc import ABC, abstractmethod
from math import ceil
from sys import stdout
from types import GeneratorType

import numpy as np
import tqdm


class Optimizer(ABC):
    def __init__(self, loss):
        self.loss = loss

    @classmethod
    def shuffle(cls, x, y):
        """ Shuffle x and y elements with the same permutation.

        :param x: an array.
        :param y: an array having the same length of x.
        :return: x, y. Permuted.
        """
        assert len(x) == len(y)
        permutation = np.random.permutation(len(x))
        return x[permutation], y[permutation]

    @abstractmethod
    def train(self, nn, x, y, **train_args):
        pass


class GradientBasedOptimizer(Optimizer):
    def forward(self, nn, x):
        """ Propagate an input signal through the network.

        :param nn: the neural network.
        :param x: the input.
        :return: two lists of length ``len(nn.layers)`` that contains,
            respectively, the input sum and the output (i.e. the result of
            applying the activation function on the input sum) at each layer.
        """
        outputs = []
        inputs = []
        xi = x

        for l in nn.layers:
            input_sum = l.input_sum(xi)
            xi = l.activation(input_sum)
            inputs.append(input_sum)
            outputs.append(xi)

        return inputs, outputs

    def backward(self, nn, x, y, inputs, outputs):
        """ Propagate an error backward through the network. See "Deep Learning"
        by Ian Goodfellow et. al (2016) p. 206.

        :param nn: the neural network.
        :param x: the input.
        :param y: the target output.
        :param inputs: a list of the input sums at each layer.
        :param outputs: a list of the activated outputs at each layer.
        :return: a list of the gradients of the loss function at each layer.
        """
        grad_weights, grad_bias = [], []  # list of size len(nn.layers)
        layers_reversed = range(len(nn.layers) - 1, -1, -1)
        g = self.loss.gradient(outputs[-1], y)

        for l in layers_reversed:
            g = np.multiply(g, nn.layers[l].activation.gradient(inputs[l]))
            grad_bias.insert(0, g)
            if l == 0:
                grad_weights.insert(0, g.dot(x.T))
            else:
                grad_weights.insert(0, g.dot(outputs[l - 1].reshape((1, -1))))
            g = nn.layers[l].weights.T.dot(g)

        return grad_weights, grad_bias

    @abstractmethod
    def train(self, nn, x, y, **train_args):
        pass


class SGD(GradientBasedOptimizer):
    def __init__(self, loss):
        super().__init__(loss)
        self.seed = None

    @staticmethod
    def batch_ranges(x, batch_size):
        """ Compute the batch ranges for a given data set.

        :param x: a list of examples.
        :param batch_size: the batch size.
        :return: a list of tuples containing the lower and the upper
            bounds (indexes) of each batch.
        """
        size = len(x)
        num_batches = int(ceil(size / float(batch_size)))
        return [(i * batch_size, min(size, (i + 1) * batch_size))
                for i in range(num_batches)]

    def train(self, nn, x, y, learning_rate=0.05, batch_size=1, epochs=1,
              metrics=None, callbacks=None, regularizer=None):
        """ Train the given neural network using the Stochastic Gradient
        Descent (SGD) algorithm.

        See "Deep Learning" by Ian Goodfellow et. al (2016) p. 286.

        :param nn: the neural network.
        :param x: a list of examples.
        :param y: the target output of each example.
        :param learning_rate: the step size of the gradient descend. It can be
            a constant, or a generator that returns the step size at each
            next() call.
        :param batch_size: the batch size.
        :param epochs: the number of the epochs.
        :param metrics: a list of metric functions to be evaluated at each
            epoch.
        :param callbacks: a list of :py:class:`nnweaver.callbacks.Callback`
            objects.
        :param regularizer: a regularizer that will be used in the training.
        """
        assert batch_size <= len(x)

        if x.ndim == 1:
            x = x.reshape((-1, 1))
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        if callbacks is None:
            callbacks = []

        np.random.seed(self.seed)
        batch_ranges = SGD.batch_ranges(x, batch_size)
        bar_format = '{l_bar}{bar}| [{elapsed}, ' '{rate_fmt}{postfix}]'
        for c in callbacks:
            c.on_training_begin(nn)

        for epoch in range(epochs):
            bar = tqdm.tqdm(batch_ranges, bar_format=bar_format, desc="Epoch %3d/%d" % (epoch + 1, epochs), file=stdout)
            x_shuffled, y_shuffled = SGD.shuffle(x, y)

            for low, high in batch_ranges:
                # Estimate gradient on the batch
                tot_errors_weight = [np.zeros(l.weights.shape) for l in nn.layers]
                tot_errors_bias = [np.zeros(l.bias.shape) for l in nn.layers]
                x_batch = x_shuffled[low:high]
                y_batch = y_shuffled[low:high]
                for i, o in zip(x_batch, y_batch):
                    i = i.reshape(-1, 1)
                    o = o.reshape(-1, 1)
                    inputs, outputs = self.forward(nn, i)
                    grad_weights, grad_bias = self.backward(nn, i, o, inputs, outputs)
                    for l in range(len(nn.layers)):
                        tot_errors_bias[l] += grad_bias[l]
                        tot_errors_weight[l] += grad_weights[l]

                # Compute the step size
                if isinstance(learning_rate, GeneratorType):
                    step = next(learning_rate)
                else:
                    step = learning_rate

                # Update weights
                for (lay, grad, grad_b) in zip(nn.layers, tot_errors_weight, tot_errors_bias):
                    assert (lay.weights - step * grad).shape == lay.weights.shape
                    weight_penalty, bias_penalty = (0, 0) if regularizer is None else regularizer.gradient(lay)
                    lay.weights -= step * (grad + weight_penalty) / batch_size
                    lay.bias -= step * (grad_b + bias_penalty) / batch_size

                bar.update(1)

            y_predicted = nn.predict_batch(x)
            loss_value = self.loss(y_predicted, y) + (0 if regularizer is None else regularizer(nn))
            metrics_values = {} if metrics is None else {m.__name__: '%.4f' % m(y_predicted, y) for m in metrics}
            bar.set_postfix(loss='%.4f' % loss_value, **metrics_values)
            bar.close()
            for c in callbacks:
                c.on_epoch_end(epoch, nn, loss_value, metrics_values)

        for c in callbacks:
            c.on_training_end(nn)


def learning_rate_time_based(initial_rate, rate_decay):
    """ Decay the rate with time.

    Formally, compute
    :math:`\\epsilon_k = \\frac{\\epsilon_{0}}{1+\\gamma k}`, where
    :math:`k` is the iteration number, :math:`\\gamma` is ``rate_decay``,
    :math:`\\epsilon_0` is ``initial_rate``.

    :param initial_rate: the rate at the first iteration.
    :param rate_decay: the decay factor.
    """
    for i in itertools.count(start=0, step=1):
        yield initial_rate / (1. + rate_decay * i)


def learning_rate_linearly_decayed(initial_rate, final_rate=0, max_iterations=20):
    """ Decay the learning rate linearly, starting from `initial_rate`. After
    `max_iterations` always yields final_rate.

    Formally, compute :math:`\\epsilon_k = (1 - \\alpha)\\epsilon_0+
    \\alpha \\epsilon_\\tau`, where :math:`k` is the iteration number,
    :math:`\\tau` is ``max_iterations``, :math:`\\epsilon_0` is
    ``initial_rate``, :math:`\\epsilon_\\tau` is ``final_rate``, and
    :math:`\\alpha=\\frac{k}{\\tau}`.


    :param initial_rate: the rate at the first iteration.
    :param final_rate: the rate to return after the maximum number of
        iterations.
    :param max_iterations: the maximum number of iterations.
    """
    for i in range(max_iterations):
        alpha = i / max_iterations
        rate = (1. - alpha) * initial_rate + alpha * final_rate
        yield rate

    while True:
        yield final_rate
