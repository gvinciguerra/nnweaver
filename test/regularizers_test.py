import numpy as np

from nnweaver import *


def test_regularizers():
    nn = NN(2)
    nn.add_layer(Layer(1))

    reg = L1L2Regularizer(1, 0)
    nn.layers[0].weights = np.matrix([[1, 0]])
    nn.layers[0].bias = np.matrix([[0]])
    assert reg(nn) == 1
    nn.layers[0].weights = np.matrix([[1, 0], [0, 0]])
    nn.layers[0].bias = np.matrix([[0], [0]])
    assert reg(nn) == 0.5
    nn.layers[0].weights = np.matrix([[1, 0], [0, 0]])
    nn.layers[0].bias = np.matrix([[0], [1]])
    assert reg(nn) == 1
    nn.layers[0].weights = np.matrix([[3, -2]])
    nn.layers[0].bias = np.matrix([[1]])
    g_weights, g_bias = reg.gradient(nn.layers[0])
    np.testing.assert_array_equal([[1, -1]], g_weights)
    np.testing.assert_array_equal([[1]], g_bias)

    reg = L1L2Regularizer(0, 1)
    nn.layers[0].weights = np.matrix([[1, 0]])
    nn.layers[0].bias = np.matrix([[0]])
    assert reg(nn) == 0.5
    nn.layers[0].weights = np.matrix([[1, 0], [0, 0]])
    nn.layers[0].bias = np.matrix([[0], [0]])
    assert reg(nn) == 0.25
    nn.layers[0].weights = np.matrix([[1, 0], [0, 0]])
    nn.layers[0].bias = np.matrix([[0], [1]])
    assert reg(nn) == 0.5
    g_weights, g_bias = reg.gradient(nn.layers[0])
    np.testing.assert_array_equal(nn.layers[0].weights, g_weights)
    np.testing.assert_array_equal(nn.layers[0].bias, g_bias)

