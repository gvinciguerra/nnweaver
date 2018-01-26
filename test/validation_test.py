from functools import partial

from nnweaver import *
from nnweaver.validation import *


def test_split_generator():
    x = y = np.arange(0, 10, 1)
    result = list(splits_generator(x, y, [4, 6]))
    np.testing.assert_array_equal(result[1][0], [4, 5, 6, 7, 8, 9])
    np.testing.assert_array_equal(result[1][3], [0, 1, 2, 3])


def test_kfold_cross_validation():
    x = np.arange(-1, 1, 0.1)
    y = np.arange(-1, 1, 0.1) ** 2

    nn = NN(1)
    nn.add_layer(Layer(12, Sigmoid))
    nn.add_layer(Layer(13, Sigmoid))
    nn.add_layer(Layer(1, Linear))
    sgd = SGD(MSE())

    k = 3
    cv_dict = kfold_cross_validation(nn, sgd, x, y, k=k, learning_rate=0.5, batch_size=5, epochs=100)
    assert 'validation_scores' in cv_dict
    assert 'train_scores' in cv_dict
    assert k == len(cv_dict['validation_scores'])
    assert k == len(cv_dict['train_scores'])
    assert np.mean(cv_dict['validation_scores']) > 0


def test_grid_search():
    x = np.arange(-1, 1, 0.1)
    y = np.arange(-1, 1, 0.1) ** 2

    def nn_builder(units_1, units_2, activation_1, activation_2):
        nn = NN(1)
        nn.add_layer(Layer(units_1, activation_1))
        nn.add_layer(Layer(units_2, activation_2))
        nn.add_layer(Layer(1, Linear))
        return nn

    sgd = SGD(MSE())
    train_args = {'epochs': [5, 10],
                  'batch_size': [10],
                  'learning_rate': [0.5]}
    builder_args = {'units_1': [5, 15],
                    'units_2': [5, 10],
                    'activation_1': [Sigmoid, Rectifier],
                    'activation_2': [Sigmoid]}

    grid_search(nn_builder, sgd, x, y, train_args, builder_args)
    two_fold = partial(kfold_cross_validation, k=2)
    grid_search(nn_builder, sgd, x, y, train_args, builder_args, two_fold)


def test_hold_out():
    x = np.arange(-1, 1, 0.1)
    y = np.arange(-1, 1, 0.1) ** 2

    nn = NN(1)
    nn.add_layer(Layer(12, Sigmoid))
    nn.add_layer(Layer(13, Sigmoid))
    nn.add_layer(Layer(1, Linear))
    sgd = SGD(MSE())

    hold_out_validation(nn, sgd, x, y, learning_rate=0.5, batch_size=5, epochs=100)
