from nnweaver.utils import *


def test_one_hot_encoding():
    one_hot = one_hot_encoding(np.array([0, 3, 2, 2], dtype=np.int8))
    expected = np.array([[1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 1, 0]])
    np.testing.assert_array_equal(one_hot, expected)
