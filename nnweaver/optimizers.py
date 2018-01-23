""" This module provides a set of optimization algorithms that can be used to
train neural networks. """

from math import ceil
from sys import stdout

import numpy as np
import tqdm


class Optimizer(object):
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

    def train(self, nn, x, y, **train_args):
        raise NotImplementedError


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

    def train(self, nn, x, y, **train_args):
        raise NotImplementedError


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
        :param learning_rate: the step size of the gradient descend.
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
        step = learning_rate / batch_size
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

                # Update weights
                for (lay, grad, grad_b) in zip(nn.layers, tot_errors_weight, tot_errors_bias):
                    assert (lay.weights - step * grad).shape == lay.weights.shape
                    weight_penalty, bias_penalty = (0, 0) if regularizer is None else regularizer.gradient(lay)
                    lay.weights -= step * (grad + weight_penalty)
                    lay.bias -= step * (grad_b + bias_penalty)

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
