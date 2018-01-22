from nnweaver.validation import *
from nnweaver import *


def test_split_generator():
    x = y = np.arange(0, 10, 1)
    result = list(splits_generator(x, y, [4, 6]))
    np.testing.assert_array_equal(result[1][0], [4, 5, 6, 7, 8, 9])
    np.testing.assert_array_equal(result[1][3], [0, 1, 2, 3])


def test_kfold_cross_validation():
    x = np.arange(-1, 1, 0.1)
    y = np.arange(-1, 1, 0.1) ** 2

    nn = NN(1)
    nn.add_layer(Layer(12, Sigmoid()))
    nn.add_layer(Layer(13, Sigmoid()))
    nn.add_layer(Layer(1, Linear()))
    sgd = SGD(MSE(), learning_rate=0.5)

    kfold_cross_validation(nn, sgd, x, y, batch_size=5, epochs=100)