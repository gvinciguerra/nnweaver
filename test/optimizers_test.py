import numpy as np

from nnweaver.activations import Linear, Sigmoid, Rectifier
from nnweaver.layer import Layer
from nnweaver.losses import MSE
from nnweaver.nn import NN
from nnweaver.optimizers import SGD, Optimizer


def test_shuffle():
    x = np.arange(0, 10, 1)
    y, z = Optimizer.shuffle(x, x)
    np.testing.assert_array_equal(y, z)


def test_sgd_bisector():
    nn = NN(1)
    nn.add_layer(Layer(1, Linear()))
    x = np.matrix(np.arange(-1, 1, 0.1)).T
    y = np.matrix(np.arange(-1, 1, 0.1)).T
    sgd = SGD(MSE(), learning_rate=0.5)
    sgd.fit(nn, x, y, 1, 100)
    np.testing.assert_almost_equal(nn.predict(-1), -1)


def test_quadratic():
    nn = NN(1)
    nn.add_layer(Layer(12, Sigmoid()))
    nn.add_layer(Layer(12, Sigmoid()))
    nn.add_layer(Layer(1, Linear()))
    x = np.matrix(np.arange(-1, 1, 0.1)).T
    y = np.matrix(np.arange(-1, 1, 0.1) ** 2).T
    sgd = SGD(MSE(), learning_rate=0.5)
    sgd.fit(nn, x, y, 5, 100)
