from nnweaver.utils import *


def test_one_hot_encoding():
    one_hot = one_hot_encoding(np.array([0, 3, 2, 2], dtype=np.int8))
    expected = np.array([[1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 1, 0]])
    np.testing.assert_array_equal(one_hot, expected)


def test_accuracy():
    a = np.array([0.7, 0.6, 0.1])
    b = np.array([1, 0, 0])
    np.testing.assert_almost_equal(accuracy(a, b), 2 / 3)

    a = np.array([0.0, 0.1, 0.4])
    b = np.array([1, 1, 1])
    np.testing.assert_almost_equal(accuracy(a, b), 0)

    a = np.array([1.0, 0.7, 0.8, 0.6])
    b = np.array([1, 1, 1, 1])
    np.testing.assert_almost_equal(accuracy(a, b), 1)

    a = np.array([1.0, 0.7])
    b = np.array([1, 0])
    np.testing.assert_almost_equal(accuracy(a, b), 1 / 2)


def test_accuracy_multiclass():
    a = np.array([[1.0, 0.6, 0.5], [0.1, 0.6, 0.3], [0.4, 0.2, 0.3], [0.1, 0.2, 0.3]])
    b = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])
    np.testing.assert_almost_equal(accuracy(a, b), 3/4)

    a = np.array([[1.0, 0.6, 0.5], [0.1, 0.6, 0.3], [0.4, 0.2, 0.3], [0.1, 0.2, 0.3]])
    b = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]])
    np.testing.assert_almost_equal(accuracy(a, b), 1 / 4)

    a = np.array([[1.0, 0.6, 0.5], [0.1, 0.6, 0.3], [0.4, 0.2, 0.3]])
    b = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
    np.testing.assert_almost_equal(accuracy(a, b), 1)
