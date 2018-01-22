import numpy as np

from nnweaver import Linear, Sigmoid, Rectifier, Layer, MSE, NN, SGD, Optimizer
from nnweaver.utils import accuracy


def test_shuffle():
    x = np.arange(0, 10, 1)
    y, z = Optimizer.shuffle(x, x)
    np.testing.assert_array_equal(y, z)


def test_sgd_bisector():
    nn = NN(1)
    nn.add_layer(Layer(1, Linear()))
    x = np.arange(-1, 1, 0.1)
    y = np.arange(-1, 1, 0.1)
    sgd = SGD(MSE())
    sgd.train(nn, x, y, 0.5, 1, 100)
    np.testing.assert_almost_equal(nn.predict(-1), -1)


def test_quadratic():
    nn = NN(1)
    nn.add_layer(Layer(12, Sigmoid()))
    nn.add_layer(Layer(13, Sigmoid()))
    nn.add_layer(Layer(1, Linear()))
    x = np.arange(-1, 1, 0.1)
    y = np.arange(-1, 1, 0.1) ** 2
    sgd = SGD(MSE())
    sgd.train(nn, x, y, 0.5, 5, 100)


def test_circle():
    samples = 50
    a = np.random.uniform(0, 2 * np.pi, samples * 2)
    r = np.append(np.random.uniform(0, 10, samples), np.random.uniform(20, 30, samples))
    x = np.matrix([np.multiply(r, np.sin(a)), np.multiply(r, np.cos(a))]).T
    y = np.append(np.ones(samples), np.zeros(samples))
    x, y = Optimizer.shuffle(x, y)
    limit = int(len(x) * 0.8)

    nn = NN(2)
    nn.add_layer(Layer(4, Rectifier()))
    nn.add_layer(Layer(1, Sigmoid()))
    sgd = SGD(MSE())
    sgd.seed = 42
    sgd.train(nn, x[:limit], y[:limit], 0.1, 5, 100, metrics=[accuracy])

    assert accuracy(nn.predict_batch(x[limit:]), y[limit:]) > 0.8
