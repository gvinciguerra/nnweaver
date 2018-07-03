""" The :py:mod:`.optimizers` module provides a set of optimization
algorithms that can be used to train neural networks.

Currently, the following optimizers are available:

1. :py:class:`.SGD` Stochastic Gradient Descent.

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
from scipy import optimize


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

    def train(self, nn, x, y, learning_rate=0.05, batch_size=1, epochs=1, momentum=0,
              metrics=None, callbacks=None, regularizer=None):
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
            bar = tqdm.tqdm(batch_ranges, bar_format=bar_format,
                            desc="Epoch %3d/%d" % (epoch + 1, epochs), file=stdout)
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
                iterator = zip(nn.layers, errors_weights,
                               errors_bias, velocity_weights, velocity_bias)
                new_velocity_bias, new_velocity_weights = [], []

                for (l, grad_w, grad_b, vel_w, vel_b) in iterator:
                    assert (l.weights - step * grad_w).shape == l.weights.shape
                    penalty_w, penalty_b = (
                        0, 0) if regularizer is None else regularizer.gradient(l)

                    # Apply penalty to weights (and compute the gradients' velocity)
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

                velocity_bias, velocity_weights = new_velocity_bias, new_velocity_weights
                bar.update(1)

            y_predicted = nn.predict_batch(x)
            loss_value = self.loss.batch_mean(
                y_predicted, y) + (0 if regularizer is None else regularizer(nn))
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


def learning_rate_linearly_decayed(initial_rate, final_rate=0, max_iterations=20):
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

    def train(self, nn, x, y, m_L=0.3, m_R=0.6, t̅=0.65, µ=1,
              accuracy_tolerance=1e-4, max_iterations=10, convex=False):
        """ Train the given neural network using the Proximal Bundle Method
        algorithm.

        :param nn: the neural network.
        :param x: a list of examples.
        :param y: a list with the target output of each example.
        :param µ: the fixed weight to be given to the
            stabilizing term throughout all the algorithm. It must be a strictly
            positive number.
        :param serious_step_condition_factor: set to a small value in [0,1) to
            prevent large steps relative to the decreasing of the loss function.
        :param accuracy_tolerance: a value greater than 0 that determines the
            stopping criterion.
        :param max_iterations: maximum number of iterations before stopping the
            training.
        """
        assert µ > 0
        assert accuracy_tolerance > 0
        assert 0 < t̅ <= 1
        assert 0 < m_L <= 0.5 and m_L < m_R < 1

        def grad_f():
            errors_bias = [np.zeros(l.bias.shape) for l in nn.layers]
            errors_weights = [np.zeros(l.weights.shape) for l in nn.layers]
            for i, o in zip(x, y):
                i = i.reshape(-1, 1)
                o = o.reshape(-1, 1)
                inputs, outputs = self.forward(nn, i)
                grad_weights, grad_bias = self.backward(
                    nn, i, o, inputs, outputs)
                for l in range(len(nn.layers)):
                    errors_bias[l] += grad_bias[l]
                    errors_weights[l] += grad_weights[l]
            return self.flatten(errors_weights, errors_bias)

        def f(w):
            self.unflatten(nn, w)
            return self.loss.batch_mean(nn.predict_batch(x), y)

        def line_search_l(ν, c, d):
            t_L = 0
            r = 1
            while r - t_L > accuracy_tolerance:
                m = (r + t_L) / 2.0
                if f(c + t_L*d) <= f(c) + m_L*t_L*ν:
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
                α = fc - fw - (t_L - t_R) * g.T.dot(d)
                if not convex:
                    α = abs(α)
                if -α + g.T.dot(d) >= m_R * ν:
                    t_R = m
                else:
                    r = m
            return t_R

        def make_constraints(c, f_c, G, F):
            constraints = []
            for gi, fi in zip(G, F):
                α = f_c - gi.dot(c) - fi
                if not convex:
                    α = abs(α)
                constr = {
                    'type': 'ineq',
                    'args': (α, gi),
                    'fun': lambda w, α, gi:
                        w[0] + α - gi.dot(w[1:].reshape((-1, 1)))
                }
                constraints.append(constr)
            return tuple(constraints)

        def bundle_objective(d, µ):
            return d[0] + µ * 0.5 * np.linalg.norm(d[1:]) ** 2.

        # Compute first function and subgradient
        list_weights = [l.weights for l in nn.layers]
        list_bias = [l.bias for l in nn.layers]

        c = self.flatten(list_weights, list_bias)
        fc = f(c)
        g = grad_f()

        G = g.T  # Matrix of subgradients
        F = np.matrix(fc - g.T.dot(c))

        for iteration in itertools.count(start=1, step=1):
            print('Iteration %d ' % iteration, end='')
            # Construct and solve the master problem
            constraints = make_constraints(c, fc, G, F)
            guess = np.vstack((np.zeros(1), c))
            res = optimize.minimize(bundle_objective, guess, args=(µ),
                                    tol=accuracy_tolerance, method='COBYLA',
                                    options={'maxiter': 1000},
                                    constraints=constraints)

            if not res.success:
                self.unflatten(nn, c)
                print('QP Solver error: ' + res.message, end='')

            d = res.x[1:]
            v = res.x[0]
            nd = np.linalg.norm(d)

            # Compute function and subgradient and update the bundle
            fd = f(c + d)
            g = grad_f()
            G = np.vstack((G, g.T))
            F = np.vstack((F, fd - g.T.dot(c + d)))

            # Serious step / null step decision
            t_L = line_search_l(v, c, d)
            if t_L >= t̅:
                c = c + t_L * d
                w = c
                print('long serious step')
            else:
                t_R = line_search_r(v, c, d, t_L)
                if t_L > 0:
                    c = c + t_L * d
                    print('short serious step')
                else:
                    print('null step')
                w = c + t_R * d

            # Stopping criteria
            if abs(v) <= accuracy_tolerance:
                print('Optimal')
                self.unflatten(nn, c)
                return 'Optimal'

            if iteration > max_iterations:
                self.unflatten(nn, c)
                return 'Exceeded max_iterations'

    @staticmethod
    def flatten(list_weights, list_bias):
        """Returns a column vector with the concatenation of the given neural
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from .nn import NN, Layer, Linear, uniform
    from .activations import Sigmoid
    from .losses import MSE

    # nn = NN(5)
    # nn.add_layer(Layer(1, Linear, bias_initializer=uniform(-0.5, 0.5)))
    # x = np.random.rand(5, 10)
    # y = 2.*x[0] + 3.*x[1] - 0.5*x[2] + x[3] - 2.*x[4]
    # pbm = ProximalBundleMethod(MSE)
    # pbm.train(nn, x.T, y.T, µ=1, accuracy_tolerance=1e-6, max_iterations=40, convex=True)
    # np.testing.assert_almost_equal(nn.predict([0, 1, 2, 3, 4]), -3, decimal=1)

    nn = NN(1)
    nn.add_layer(
        Layer(6, Sigmoid, weights_initializer=uniform(-0.0005, 0.000005)))
    nn.add_layer(
        Layer(1, Linear, weights_initializer=uniform(-0.0005, 0.000005)))
    x = np.arange(-10, 10)
    y = x ** 2.0
    y = -10 - 10 * Sigmoid.apply(x)
    pbm = ProximalBundleMethod(MSE)
    pbm.train(nn, x.T, y.T, µ=0.5, accuracy_tolerance=1e-3, max_iterations=30)
    # np.testing.assert_almost_equal(nn.predict([0, 1, 2, 3, 4]), -3, decimal=1)
    plt.scatter(x, y, label='dataset')
    plt.scatter(x, nn.predict_batch(x.T), label='nn')
    plt.legend()
    plt.show()
