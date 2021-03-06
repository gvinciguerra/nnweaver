import numpy as np

from nnweaver.activations import Linear
from nnweaver import Layer


def test_call():
    layer = Layer(3, Linear)
    layer.weights = np.eye(3)
    layer.bias = np.ones(3)
    v = np.array([1, 2, 3])
    np.testing.assert_array_equal(layer(v), np.array([2, 3, 4]))

