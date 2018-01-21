import numpy as np

from activations import Sigmoid, Rectifier, Linear


def test_linear():
    v = np.array([1, 2, 3])
    l = Linear()
    np.testing.assert_array_almost_equal(l(v), v)
    np.testing.assert_array_almost_equal(l.gradient(v), np.ones(3))


def test_sigmoid():
    v = np.array([1, 2, 3])
    s = Sigmoid()
    np.testing.assert_array_almost_equal(s(v), np.array([0.731059, 0.880797, 0.952574]), decimal=6)
    np.testing.assert_array_almost_equal(s.gradient(v), np.array([0.196612, 0.104994, 0.045177]), decimal=6)


def test_rectifier():
    v = np.array([1, -2, 3])
    r = Rectifier()
    np.testing.assert_array_almost_equal(r(v), np.array([1, 0, 3]))
    np.testing.assert_array_almost_equal(r.gradient(v), np.array([1, 0, 1]))