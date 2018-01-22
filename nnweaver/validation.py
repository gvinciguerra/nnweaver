import itertools
import operator
import time
from functools import reduce
from typing import Callable, Tuple

import numpy as np

from nnweaver import NN
from nnweaver.optimizers import Optimizer


def splits_generator(x, y, groups):
    """ Split two data sets of the same size into partitions of the given dimensions.

    :param x: a list of elements.
    :param y: a list of elements.
    :param groups: the list of the sizes of each partition (the sum of its elements must be
        ``len(x)``.
    """
    assert len(x) == len(y)
    assert sum(groups) == len(x)

    begin = 0
    for g in groups:
        end = begin + g
        yield x[begin:end], y[begin:end], \
              np.concatenate((x[:begin], x[end:])), \
              np.concatenate((y[:begin], y[end:]))
        begin += g


def kfold_cross_validation(nn, optimizer, x, y, k=3, **train_args):
    """ Perform a K-Fold Cross Validation of the given neural network.

    It splits the data set into :math:`k` partitions that will be used to test :math:`k` different
    models (trained on the other :math:`k - 1` partitions) and then returns the best performing one.

    :param nn: a neural network.
    :param optimizer: the optimizer used to train the neural network. Its signature
        must be compatible with the keys in ``train_args``.
    :param x: a list of examples.
    :param y: the target output of each example.
    :param k: the number of partitions (folds) of the data set.
    :param train_args: a dictionary whose keys are compatible with the arguments of
        ``optimizer.train()``.
    :return: the best (of the :math:`k`) model and the value of the its loss obtained on the test
        partition.
    """
    split_size = int(len(x) / k)
    groups = [split_size for _ in range(k - 1)] + [len(x) - split_size * (k - 1)]
    best_loss = np.inf
    best_model = None
    fold = 1

    for split_x, split_y, split_complement_x, split_complement_y in splits_generator(x, y, groups):
        nn_i = nn.clone()
        nn_i.reset()
        t_start = time.time()
        optimizer.train(nn, split_complement_x, split_complement_y, **train_args)
        elapsed = (time.time() - t_start) * 1000
        loss_value = optimizer.loss(nn_i.predict_batch(x), y)
        if loss_value < best_loss:
            best_model = nn_i
            best_loss = loss_value

        print('📂 Fold %d/%d completed in %dms, best loss so far %.4f.' % (fold, k, elapsed, best_loss))
        fold += 1

    return best_model, best_loss


def grid_search(nn_builder: Callable[[dict], NN],
                optimizer, x, y, train_args, builder_args,
                cv: Callable[[NN, Optimizer, object, object], Tuple[NN, float]] = None):
    """ Perform an exhaustive search through a space of training and neural
    network topology parameters.

    An *iteration* of the grid search is one of the possible combinations of
    the training and builder parameters. The first ones will be fed to
    ``optimizer.train()``, while the second ones will be fed to ``nn_builder``.

    :param nn_builder: a function that accepts a parameter ``builder_args`` and
        returns a neural network.
    :param optimizer: the optimizer to use in the grid search. Its signature
        must be compatible with the keys in ``train_args``.
    :param x: a list of examples.
    :param y: the target output of each example.
    :param train_args: a dictionary whose keys are:
        (1) compatible with the arguments of ``optimizer.train()``, and
        (2) associated with lists that represent the subset of the arguments to
        explore.
    :param builder_args: a dictionary whose keys are:
        (1) compatible with the arguments of ``nn_builder()``, and
        (2) associated with lists that represent the subset of arguments to
        explore.
    :param cv: an optional cross validation method that is called at each iteration.
    :return: the best model found by the grid search and the loss value.
    """
    train_args_keys, train_args_values = train_args.keys(), list(train_args.values())
    builder_args_keys, builder_args_values = builder_args.keys(), list(builder_args.values())
    best_loss = np.inf
    best_model = None

    tot_iterations = reduce(operator.mul, map(len, train_args_values + builder_args_values))
    print('Beginning grid search with %d iterations' % tot_iterations)

    for t in itertools.product(*train_args_values):
        for b in itertools.product(*builder_args_values):
            t_args = dict(zip(train_args_keys, t))
            b_args = dict(zip(builder_args_keys, b))
            nn = nn_builder(**b_args)
            if cv is None:
                optimizer.train(nn, x, y, **t_args)
                loss_value = optimizer.loss(nn.predict_batch(x), y)
            else:
                nn, loss_value = cv(nn, optimizer, x, y, **t_args)
            if loss_value < best_loss:
                best_loss = loss_value
                best_model = nn

    return best_model, best_loss


def hold_out_validation(nn, optimizer, x, y, train_ratio=0.8, **train_args):
    """ Perform an Hold-Out Validation of the given neural network.

    :param nn: a neural network.
    :param optimizer: the optimizer used to train the neural network. Its signature
        must be compatible with the keys in ``train_args``.
    :param x: a list of examples.
    :param y: the target outputo of each example.
    :param train_ratio: the ratio between the size of the partition of examples used to
        train the neural network and the one used to test it.
    :param train_args: a dictionary whose keys are compatible with the arguments of
        ``optimizer.train()``.
    :return: the trained model and the value of the loss computed on the test partition.
    """
    assert 0 < train_ratio < 1
    train_size = int(len(x)*train_ratio)
    test_size = len(x) - train_size
    train_x, train_y, test_x, test_y = next(splits_generator(x, y, [train_size, test_size]))
    optimizer.train(nn, train_x, train_y, **train_args)

    return nn, optimizer.loss(nn.predict_batch(test_x), test_y)
