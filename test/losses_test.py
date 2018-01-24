from nnweaver.losses import *


def test_mee():
    x = np.array(1)
    y = np.array(2)
    np.testing.assert_almost_equal(MEE.loss(x, y), 1)

    x = np.array([[1], [2]])
    y = np.array([[3], [5]])
    np.testing.assert_almost_equal(MEE.loss(x, y), np.sqrt(13))

    x = np.array([[1, 3, 4], [2, 3, 4]])
    y = np.array([[3, 4, 5], [5, 2, 1]])
    expect = np.array([np.sqrt(6), np.sqrt(19)]).mean()
    np.testing.assert_almost_equal(MEE.batch_mean(x, y), expect)


def test_mse():
    x = np.array(1)
    y = np.array(2)
    np.testing.assert_almost_equal(MSE.loss(x, y), 0.5)

    x = np.array([[1], [2]])
    y = np.array([[3], [5]])
    np.testing.assert_almost_equal(MSE.loss(x, y), 13. / 2)

    x = np.array([[1, 3, 4], [2, 3, 4]])
    y = np.array([[3, 4, 5], [5, 2, 1]])
    expect = np.array([6. / 2, 19. / 2]).mean()
    np.testing.assert_almost_equal(MSE.batch_mean(x, y), expect)
