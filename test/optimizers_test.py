import numpy as np

from nnweaver import *
from nnweaver.optimizers import Optimizer
from nnweaver.utils import accuracy


def test_shuffle():
    x = np.arange(0, 10, 1)
    y, z = Optimizer.shuffle(x, x)
    np.testing.assert_array_equal(y, z)


def test_sgd_bisector():
    x = np.arange(-1, 1, 0.1)
    y = np.arange(-1, 1, 0.1)
    nn = NN(1)
    nn.add_layer(Layer(1, Linear))
    sgd = SGD(MSE())
    sgd.train(nn, x, y, 0.5, 1, 100)
    np.testing.assert_almost_equal(nn.predict(-1), -1)


def test_sgd_multioutput():
    x = np.arange(-1, 1.1, 0.1)
    y = np.stack((np.arange(-1, 1.1, 0.1), np.arange(-1, 1.1, 0.1)), axis=-1)
    nn = NN(1)
    nn.add_layer(Layer(2, Linear))
    sgd = SGD(MSE())
    sgd.train(nn, x, y, 0.5, 1, 100)
    np.testing.assert_almost_equal(nn.predict(-0.2), np.matrix([-0.2, -0.2]).T)


def test_sgd_quadratic():
    nn = NN(1)
    nn.add_layer(Layer(12, Sigmoid))
    nn.add_layer(Layer(13, Sigmoid))
    nn.add_layer(Layer(1, Linear))
    x = np.arange(-1, 1, 0.1)
    y = np.arange(-1, 1, 0.1) ** 2
    sgd = SGD(MSE())
    sgd.train(nn, x, y, 0.5, 5, 100)


def test_sgd_circles():
    samples = 50
    a = np.random.uniform(0, 2 * np.pi, samples * 2)
    r = np.append(np.random.uniform(0, 10, samples),
                  np.random.uniform(20, 30, samples))
    x = np.matrix([np.multiply(r, np.sin(a)), np.multiply(r, np.cos(a))]).T
    y = np.append(np.ones(samples), np.zeros(samples))
    x, y = Optimizer.shuffle(x, y)
    limit = int(len(x) * 0.8)

    nn = NN(2)
    nn.add_layer(Layer(4, Rectifier))
    nn.add_layer(Layer(1, Sigmoid))
    sgd = SGD(MSE())
    sgd.seed = 42
    sgd.train(nn, x[:limit], y[:limit], 0.1, 5, 100, metrics=[accuracy])
    assert accuracy(nn.predict_batch(x[limit:]), y[limit:]) > 0.7


def test_learning_rate():
    l0 = learning_rate_time_based(0.5, 0.1)
    np.testing.assert_almost_equal(next(l0), 0.5)
    np.testing.assert_almost_equal(next(l0), 0.4545454545)

    l1 = learning_rate_linearly_decayed(0.5, 0.2, 3)
    np.testing.assert_almost_equal(next(l1), 0.5)
    np.testing.assert_almost_equal(next(l1), 0.4)
    np.testing.assert_almost_equal(next(l1), 0.3)
    np.testing.assert_almost_equal(next(l1), 0.2)
    np.testing.assert_almost_equal(next(l1), 0.2)


def test_bundle_flatten():
    nn = NN(3)
    nn.add_layer(Layer(2, Linear))
    nn.add_layer(Layer(3, Linear))
    cloned = nn.clone()
    cloned.reset()
    w = ProximalBundleMethod.flatten([l.weights for l in nn.layers],
                                     [l.bias for l in nn.layers])
    ProximalBundleMethod.unflatten(cloned, w)
    for l1, l2 in zip(nn.layers, cloned.layers):
        np.testing.assert_almost_equal(l2.bias, l1.bias)
        np.testing.assert_almost_equal(l2.weights, l1.weights)


def test_bundle_bisector():
    x = np.arange(-1, 1, 0.2)
    y = np.arange(-1, 1, 0.2)
    nn = NN(1)
    nn.add_layer(Layer(1, Linear, he_normal(), he_normal()))
    pbm = ProximalBundleMethod(MSE)
    pbm.train(nn, x, y,  mu=0.001, m_L=0.3, m_R=0.7, t_bar=0.5, gamma=0,
              accuracy_tolerance=1e-10, max_iterations=500)
    np.testing.assert_almost_equal(nn.predict(-1), -1, decimal=3)


def test_bundle_linear():
    nn = NN(3)
    nn.add_layer(Layer(1, Linear, glorot_uniform(), uniform(0, 0)))
    x = np.array(
        [[0.14131787,  0.31549032,  0.33582581,  0.16351758,  0.23220519],
         [0.34221643,  0.36613729,  0.9500988,  0.74681656,  0.08620996],
         [0.69490933,  0.05386328,  0.08184282,  0.83205677,  0.6950512]])
    y = 2.*x[0] + 3.*x[1] - 0.5*x[2]
    pbm = ProximalBundleMethod(MSE)
    pbm.train(nn, x.T, y.T, mu=0.001, m_L=0.3, m_R=0.7, t_bar=0.5, gamma=0,
              accuracy_tolerance=1e-10, max_iterations=500)
    np.testing.assert_almost_equal(nn.predict([0, 1, 2]), 2, decimal=3)
