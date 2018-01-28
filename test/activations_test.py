import numpy as np

from activations import Sigmoid, Rectifier, Linear, TanH, HardSigmoid, HardTanH


def test_linear():
    v = np.array([1, 2, 3])
    np.testing.assert_array_almost_equal(Linear.apply(v), v)
    np.testing.assert_array_almost_equal(Linear.gradient(v), np.ones(3))


def test_sigmoid():
    v = np.array([1, 2, 3])
    np.testing.assert_array_almost_equal(Sigmoid.apply(v), np.array([0.731059, 0.880797, 0.952574]), decimal=6)
    np.testing.assert_array_almost_equal(Sigmoid.gradient(v), np.array([0.196612, 0.104994, 0.045177]), decimal=6)


def test_rectifier():
    v = np.array([1, -2, 3])
    np.testing.assert_array_almost_equal(Rectifier.apply(v), np.array([1, 0, 3]))
    np.testing.assert_array_almost_equal(Rectifier.gradient(v), np.array([1, 0, 1]))


def test_tanh():
    v = np.array([1, 2, 3])
    np.testing.assert_array_almost_equal(TanH.apply(v), np.array([0.761594, 0.964028, 0.995055]))
    np.testing.assert_array_almost_equal(TanH.gradient(v), np.array([0.419974, 0.0706508, 0.00986604]))


def test_hard_sigmoid():
    v = np.array([-3, -2.5, -1, 0, 1, 2.5, 3])
    np.testing.assert_array_almost_equal(HardSigmoid.apply(v), np.array([0., 0., 0.3, 0.5, 0.7, 1., 1.]))
    np.testing.assert_array_almost_equal(HardSigmoid.gradient(v), np.array([0., 0., 0.2, 0.2, 0.2, 0., 0.]))


def test_hard_tanh():
    v = np.array([-3, -2.5, -1, 0, 1, 2.5, 3])
    np.testing.assert_array_almost_equal(HardTanH.apply(v), np.array([-1., -1., -1., 0, 1., 1., 1.]))
    np.testing.assert_array_almost_equal(HardTanH.gradient(v), np.array([0., 0., 0., 1., 0., 0., 0.]))
