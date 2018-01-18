import numpy as np

from nnweaver.activations import Linear, Sigmoid, Rectifier
from nnweaver.layer import Layer
from nnweaver.losses import MSE
from nnweaver.nn import NN
from nnweaver.optimizers import SGD


def test_sgd_bisector():
    nn = NN(1)
    nn.add_layer(Layer(1, Linear()))
    X = np.matrix(np.arange(-1, 1, 0.1)).T
    Y = np.matrix(np.arange(-1, 1, 0.1)).T
    sgd = SGD(MSE(), learning_rate=0.5)
    sgd.fit(nn, X, Y, 1, 100)
    np.testing.assert_almost_equal(nn.predict(-1), -1)


def test_quadratic():
    nn = NN(1)
    nn.add_layer(Layer(12, Sigmoid()))
    nn.add_layer(Layer(12, Sigmoid()))
    nn.add_layer(Layer(1, Linear()))
    X = np.matrix(np.arange(-1, 1, 0.1)).T
    Y = np.matrix(np.arange(-1, 1, 0.1) ** 2).T
    sgd = SGD(MSE(), learning_rate=0.5)
    sgd.fit(nn, X, Y, 5, 100)
