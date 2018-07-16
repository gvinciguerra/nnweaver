""" The :py:mod:`.optimizers` module provides a set of optimization
algorithms that can be used to train neural networks.

Currently, the following optimizers are available:

1. :py:class:`.SGD` Stochastic Gradient Descent.
2. :py:class:`.ProximalBundleMethod` Proximal Bundle Method.

"""

import functools
import itertools
import operator
from abc import ABC, abstractmethod
from math import ceil
from sys import stdout
from types import GeneratorType

import numpy as np
import tqdm
from cvxopt import matrix, solvers


class Optimizer(ABC):
    @abstractmethod
    def __init__(self, loss):
        """ Abstract base class for optimizers.

        :param loss: the loss function to optimize.
        """
        self.loss = loss

    @classmethod
    def shuffle(cls, x, y):
        """ Shuffle x and y elements with the same permutation.

        :param x: a list of examples.
        :param y: a list with the target output of each example.
        :return: x, y. Permuted.
        """
        assert len(x) == len(y)
        permutation = np.random.permutation(len(x))
        return x[permutation], y[permutation]

    @abstractmethod
    def train(self, nn, x, y, **train_args):
        pass


class GradientBasedOptimizer(Optimizer):
    @abstractmethod
    def __init__(self, loss):
        """ Abstract class for optimizers that use the gradient information.

        :param loss: the loss function to optimize.
        """
        super().__init__(loss)

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
            xi = l.activation.apply(input_sum)
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
        """ Create an optimizer that implement the Stochastic Gradient Descent
        (SGD) algorithm.

        :param loss: the loss (cost) function to optimize.
        """
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
              momentum=0, metrics=None, callbacks=None, regularizer=None):
        """ Train the given neural network using the Stochastic Gradient
        Descent (SGD) algorithm.

        See "Deep Learning" by Ian Goodfellow et. al (2016) p. 286.

        :param nn: the neural network.
        :param x: a list of examples.
        :param y: a list with the target output of each example.
        :param learning_rate: the step size of the gradient descend. It can be
            a constant, or a generator that returns the step size at each
            next() call.
        :param batch_size: the batch size.
        :param epochs: the number of the epochs.
        :param momentum: value between 0 and 1. Determines how quickly the
            contribution of previous gradients exponentially decay.
        :param metrics: a list of metric functions to be evaluated at each
            epoch.
        :param callbacks: a list of :py:class:`.Callback`
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
        velocity_bias = np.zeros(len(nn.layers))
        velocity_weights = np.zeros(len(nn.layers))
        for c in callbacks:
            c.on_training_begin(nn)

        for epoch in range(epochs):
            bar = tqdm.tqdm(batch_ranges, bar_format=bar_format, file=stdout,
                            desc="Epoch %3d/%d" % (epoch + 1, epochs))
            x_shuffled, y_shuffled = SGD.shuffle(x, y)

            for low, high in batch_ranges:
                # Estimate gradient on the batch
                errors_bias = [np.zeros(l.bias.shape) for l in nn.layers]
                errors_weights = [np.zeros(l.weights.shape) for l in nn.layers]
                x_batch = x_shuffled[low:high]
                y_batch = y_shuffled[low:high]
                for i, o in zip(x_batch, y_batch):
                    i = i.reshape(-1, 1)
                    o = o.reshape(-1, 1)
                    inputs, outputs = self.forward(nn, i)
                    grad_weights, grad_bias = self.backward(
                        nn, i, o, inputs, outputs)
                    for l in range(len(nn.layers)):
                        errors_bias[l] += grad_bias[l]
                        errors_weights[l] += grad_weights[l]

                # Compute the step size
                if isinstance(learning_rate, GeneratorType):
                    step = next(learning_rate)
                else:
                    step = learning_rate

                # Update weights
                self._weights_gradient = (errors_weights, errors_bias)
                iterator = zip(nn.layers, errors_weights,
                               errors_bias, velocity_weights, velocity_bias)
                new_velocity_bias, new_velocity_weights = [], []

                for (l, grad_w, grad_b, vel_w, vel_b) in iterator:
                    assert (l.weights - step * grad_w).shape == l.weights.shape
                    penalty_w, penalty_b = (0, 0) if regularizer is None \
                        else regularizer.gradient(l)

                    # Apply penalty to weights
                    g_w = step * (grad_w + penalty_w) / batch_size
                    g_b = step * (grad_b + penalty_b) / batch_size

                    # Compute velocity update
                    v_b = g_b + momentum * vel_b
                    v_w = g_w + momentum * vel_w
                    new_velocity_bias.append(vel_b)
                    new_velocity_weights.append(vel_w)

                    # Apply update
                    l.bias -= v_b
                    l.weights -= v_w

                velocity_bias = new_velocity_bias
                velocity_weights = new_velocity_weights
                bar.update(1)

            y_predicted = nn.predict_batch(x)
            loss_value = self.loss.batch_mean(y_predicted, y) \
                + (0 if regularizer is None else regularizer(nn))
            metrics_values = {} if metrics is None else {
                m.__name__: '%.4f' % m(y_predicted, y) for m in metrics}
            bar.set_postfix(loss='%.4f' % loss_value, **metrics_values)
            bar.close()
            for c in callbacks:
                c.on_epoch_end(epoch, nn, loss_value, metrics_values)

        for c in callbacks:
            c.on_training_end(nn)


def learning_rate_time_based(initial_rate, rate_decay):
    """ A generator function that decays the rate with time.

    It can be passed as ``learning_rate`` argument to :py:meth:`.SGD.train` .

    Formally, it computes
    :math:`\\epsilon_k = \\frac{\\epsilon_{0}}{1+\\gamma k}`, where
    :math:`k` is the iteration number, :math:`\\gamma` is ``rate_decay``,
    :math:`\\epsilon_0` is ``initial_rate``.

    :param initial_rate: the rate at the first iteration.
    :param rate_decay: the decay factor.
    """
    for i in itertools.count(start=0, step=1):
        yield initial_rate / (1. + rate_decay * i)


def learning_rate_linearly_decayed(initial_rate, final_rate=0,
                                   max_iterations=20):
    """ A generator function that, starting from `initial_rate`, decays the
    learning rate linearly. After `max_iterations` it always yields final_rate.

    It can be passed as ``learning_rate`` argument to :py:meth:`.SGD.train` .

    Formally, it computes :math:`\\epsilon_k = (1 - \\alpha)\\epsilon_0+
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


class ProximalBundleMethod(GradientBasedOptimizer):
    def __init__(self, loss):
        """ Create an optimizer that implement the Proximal Bundle Method
        algorithm.

        :param loss: the loss (cost) function to optimize.
        """
        super().__init__(loss)
        self.seed = None

    def train(self, nn, x, y, mu=1., m_L=0.1, m_R=0.99, t_bar=0.5, gamma=0.5,
              accuracy_tolerance=1e-4, max_iterations=10, a_bar=np.inf,
              regularizer=None, callbacks=None):
        """ Train the given neural network using the Proximal Bundle Method
        algorithm.

        :param nn: the neural network.
        :param x: a list of examples.
        :param y: a list with the target output of each example.
        :param mu: the fixed weight to be given to the
            stabilizing term throughout all the algorithm. It must be a
            strictly positive number.
        :param m_L: line search parameter. Must be 0 < m_L <= 0.5.
        :param m_R: line search parameter. Must be m_L < m_R < 1.
        :param t_bar: set to a small value in (0,1) to prevent large steps
            relative to the decreasing of the loss function.
        :param gamma: the distance measure parameter. Higher values lead to
            more localized information of the subgradients. Must be γ >= 0.
        :param accuracy_tolerance: a value greater than 0 that determines the
            stopping criterion.
        :param max_iterations: maximum number of iterations before stopping the
            training.
        :param a_bar: the locality radius tolerance. The subgradient
            information collected outside the ball of radius a_bar around
            the current solution will be discarded.
        :param regularizer: a regularizer that will be used in the training.
        :param callbacks: a list of :py:class:`.Callback` objects.
        """
        t̅ = t_bar
        µ = mu
        γ = gamma
        a̅ = a_bar

        assert µ > 0
        assert accuracy_tolerance > 0
        assert 0 < t̅ <= 1
        assert 0 < m_L <= 0.5 and m_L < m_R < 1
        assert γ >= 0
        assert a̅ >= 0

        if callbacks is None:
            callbacks = []

        def grad_f():
            errors_bias = [np.zeros(l.bias.shape) for l in nn.layers]
            errors_weights = [np.zeros(l.weights.shape) for l in nn.layers]
            for i, o in zip(x, y):
                i = i.reshape(-1, 1)
                o = o.reshape(-1, 1)
                inputs, outputs = self.forward(nn, i)
                grad_weights, grad_bias = self.backward(nn, i, o, inputs,
                                                        outputs)
                for l in range(len(nn.layers)):
                    weight_penalty, bias_penalty = (0, 0)
                    if regularizer is not None:
                        weight_penalty, bias_penalty = regularizer.gradient(
                            nn.layers[l])
                    errors_bias[l] += (grad_bias[l] + bias_penalty) / len(x)
                    errors_weights[l] += (grad_weights[l] + weight_penalty) / len(x)
            return self.flatten(errors_weights, errors_bias)

        def f(θ):
            self.unflatten(nn, θ)
            penalty = 0 if regularizer is None else regularizer(nn)
            return self.loss.batch_mean(nn.predict_batch(x), y) + penalty

        def line_search_l(ν, c, d):
            t_L = 0
            r = 1
            while r - t_L > accuracy_tolerance:
                m = (r + t_L) / 2.0
                if f(c + t_L * d) <= f(c) + m_L * t_L * ν:
                    t_L = m
                else:
                    r = m
            return t_L

        def line_search_r(ν, c, d, t_L):
            t_R = t_L
            r = 1
            while r - t_R > accuracy_tolerance:
                m = (r + t_R) / 2.0
                fc = f(c + t_L * d)
                fw = f(c + t_R * d)
                g = grad_f()
                α = abs(fc - fw - (t_L - t_R) * g.T.dot(d))
                if -α + g.T.dot(d) >= m_R * ν:
                    t_R = m
                else:
                    r = m
            return t_R

        def make_parameters(c, f_c, G, F, S):
            α = f_c - G.dot(c) - F
            h = np.maximum(np.abs(α), γ * S ** 2)
            A = np.hstack((-np.ones((G.shape[0], 1)), G))
            P = µ * np.eye(len(c) + 1)
            P[0, 0] = 0
            q = np.eye(len(c) + 1, 1)
            return matrix(P), matrix(q), matrix(A), matrix(h)

        # Compute first function and subgradient
        list_weights = [l.weights for l in nn.layers]
        list_bias = [l.bias for l in nn.layers]

        θ = self.flatten(list_weights, list_bias)
        c = θ
        fc = f(c)
        g = grad_f()
        a = 0

        G = g.T  # Matrix of subgradients
        F = np.matrix(fc - g.T.dot(c))
        S = np.zeros(1)  # Info to compute the subgrad locality measure

        for clbk in callbacks:
            clbk.on_training_begin(nn)

        bar = tqdm.tqdm(total=max_iterations)
        for iteration in itertools.count(start=1, step=1):
            bar.write('Iteration %3d: ' % iteration, end='')
            res = solvers.qp(*make_parameters(c, fc, G, F, S),
                             options={'show_progress': False})

            if res['status'] is not 'optimal':
                self.unflatten(nn, c)
                bar.write('QP Solver error: ' + res['status'], end='')

            d = res['x'][1:]
            ν = res['x'][0]

            # Compute function and subgradient and update the bundle
            fd = f(θ)
            g = grad_f()
            fc = f(c)
            G = np.vstack((G, g.T))
            F = np.vstack((F, fd - g.T.dot(θ)))

            # Serious step / null step decision
            t_L = line_search_l(ν, c, d)
            d_norm = np.linalg.norm(d)
            if t_L >= t̅:
                bar.write('Long Serious Step, ', end='')
                c = c + t_L * d
                θ = c
                s_c = t_L * d_norm
                s_d = 0
            else:
                t_R = line_search_r(ν, c, d, t_L)
                if t_L > 0:
                    bar.write('Short Serious Step, ', end='')
                    c = c + t_L * d
                    θ = c + t_R * d
                    s_c = t_L * d_norm
                    s_d = (t_R - t_L) * d_norm
                else:
                    bar.write('Null Step, ', end='')
                    θ = c + t_R * d
                    s_c = 0
                    s_d = t_R * d_norm
            S += s_c
            S = np.vstack((S, s_d))
            a = max(a + s_c, s_d)

            if a > a̅:
                mask = (S < 0.5 * a̅).flatten()
                mask[-1] = True  # Ensures at least one element
                G = G[mask]
                F = F[mask]
                S = S[mask]
                a = S[0]

            bar.write('{} constraints'.format(len(F)))

            # Stopping criteria
            self.unflatten(nn, c)
            if abs(ν) <= accuracy_tolerance:
                bar.write('Optimal')
                break
            if iteration > max_iterations:
                bar.write('Exceeded max_iterations')
                break

            # Callbacks
            y_predicted = nn.predict_batch(x)
            loss_value = self.loss.batch_mean(y_predicted, y)
            for clbk in callbacks:
                clbk.on_epoch_end(iteration, nn, loss_value, {})
            bar.set_postfix({'loss': loss_value, '‖d‖': d_norm, 'ν': ν})
            bar.update()

        for clbk in callbacks:
            clbk.on_training_end(nn)

    @staticmethod
    def flatten(list_weights, list_bias):
        """ Returns a column vector with the concatenation of the given neural
        network parameters. """
        flattened = np.array([])
        for w, b in zip(list_weights, list_bias):
            flattened = np.concatenate((flattened, w.flatten(), b.flatten()))
        return flattened.reshape((-1, 1))

    @staticmethod
    def unflatten(nn, flattened):
        """ Plug back the flattened weights/bias into the neural networks. """
        low = 0
        for l in nn.layers:
            high = low + functools.reduce(operator.mul, l.weights.shape, 1)
            l.weights = flattened[low:high].reshape(l.weights.shape)
            low = high
            high = low + l.bias.shape[0]
            l.bias = flattened[low:high].reshape(l.bias.shape)
            low = high
