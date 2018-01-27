""" The :py:mod:`.validation` module provides a set of cross validation
and model selection methods. """
import inspect
import itertools
import operator
import time
from functools import reduce
from typing import Callable, Dict, List, Union, Any

import numpy as np
from scipy.stats.distributions import rv_frozen

from nnweaver import NN
from nnweaver.optimizers import Optimizer


def splits_generator(x, y, groups):
    """ Split a data set into partitions.

    :param x: a list of examples.
    :param y: a list with the target output of each example.
    :param groups: the list of the sizes of each partition (the sum of its
        elements must be ``len(x)``.
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

    It splits the data set into ``k`` partitions that will be used to test
    ``k`` different models (trained on the other ``k`` - 1 partitions).

    :param nn: a neural network.
    :param optimizer: the optimizer used to train the neural network. Its
        signature must be compatible with the keys in ``train_args``.
    :param x: a list of examples.
    :param y: a list with the target output of each example.
    :param k: the number of partitions (folds) of the data set.
    :param train_args: a dictionary whose keys are compatible with the arguments
        of ``optimizer.train()``.
    :return: a dictionary with keys ``'validation_scores'`` and
        ``'train_scores'`` whose values are arrays of size ``k``.
    """
    split_size = int(len(x) / k)
    groups = [split_size for _ in range(k - 1)] + [len(x) - split_size * (k - 1)]
    train_scores = []
    validation_scores = []
    fold = 1

    for split_x, split_y, split_complement_x, split_complement_y in splits_generator(x, y, groups):
        nn_i = nn.clone()
        nn_i.reset()
        t_start = time.time()
        optimizer.train(nn, split_complement_x, split_complement_y, **train_args)
        elapsed = (time.time() - t_start) * 1000

        train_loss = optimizer.loss.batch_mean(nn_i.predict_batch(split_complement_x), split_complement_y)
        validation_loss = optimizer.loss.batch_mean(nn_i.predict_batch(x), y)
        train_scores.append(train_loss)
        validation_scores.append(validation_loss)

        print('📂 Fold %d/%d completed in %dms' % (fold, k, elapsed))
        fold += 1

    return {'validation_scores': validation_scores, 'train_scores': train_scores}


def hold_out_validation(nn, optimizer, x, y, train_ratio=0.8, **train_args):
    """ Perform an Hold-Out Validation of the given neural network.

    :param nn: a neural network.
    :param optimizer: the optimizer used to train the neural network. Its
        signature must be compatible with the keys in ``train_args``.
    :param x: a list of examples.
    :param y: a list with the target output of each example.
    :param train_ratio: the ratio between the size of the partition of examples
        used to train the neural network and the one used to test it.
    :param train_args: a dictionary whose keys are compatible with the arguments
        of ``optimizer.train()``.
    :return: the trained model and the value of the loss computed on the test
        partition.
    """
    assert 0 < train_ratio < 1
    train_size = int(len(x) * train_ratio)
    test_size = len(x) - train_size
    train_x, train_y, test_x, test_y = next(splits_generator(x, y, [train_size, test_size]))
    optimizer.train(nn, train_x, train_y, **train_args)
    train_loss = optimizer.loss.batch_mean(nn.predict_batch(train_x), train_y)
    validation_loss = optimizer.loss.batch_mean(nn.predict_batch(test_x), test_y)

    return {'validation_scores': [validation_loss], 'train_scores': [train_loss]}


def __check_search_args_compatibility(nn_builder, optimizer, train_args, builder_args):
    """ This is an utility function used in :py:meth:`.grid_search` and
    :py:meth:`.random_search` to check whether or not the hyperparameters
    given by the user are compatible.
    """
    train_args_keys = set(train_args.keys())
    builder_args_keys = set(builder_args.keys())
    train_args = set(inspect.getfullargspec(optimizer.train)[0])
    nn_builder_args = set(inspect.getfullargspec(nn_builder)[0])
    diff_train_args = train_args_keys.difference(train_args)
    diff_builder_args = builder_args_keys.difference(nn_builder_args)
    assert len(diff_train_args) == 0, 'Arguments %s cannot applied to train()' % diff_train_args
    assert len(diff_builder_args) == 0, 'Arguments %s cannot applied to nn_builder' % diff_builder_args


def grid_search(nn_builder: Callable[[dict], NN],
                optimizer, x, y, train_args, builder_args,
                cv: Callable[[NN, Optimizer, object, object], Dict[str, List[float]]] = None):
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
    :param y: a list with the target output of each example.
    :param train_args: a dictionary whose keys are:
        (1) compatible with the arguments of ``optimizer.train()``, and
        (2) associated with lists that represent the subset of the arguments to
        explore.
    :param builder_args: a dictionary whose keys are:
        (1) compatible with the arguments of ``nn_builder()``, and
        (2) associated with lists that represent the subset of arguments to
        explore.
    :param cv: an optional cross validation method that is called at each
        iteration.
    :return: a tuple with:
        (1) the best model found by the grid search,
        (2) its loss value,
        (3) arguments used to train it, and
        (4) arguments used to build it.
    """
    __check_search_args_compatibility(nn_builder, optimizer, train_args, builder_args)
    train_args_keys, train_args_values = train_args.keys(), list(train_args.values())
    builder_args_keys, builder_args_values = builder_args.keys(), list(builder_args.values())
    best_loss = np.inf
    best_model = None
    best_train_args = None
    best_builder_args = None

    tot_iterations = reduce(operator.mul, map(len, train_args_values + builder_args_values))
    print('Beginning grid search with %d iterations' % tot_iterations)

    for t in itertools.product(*train_args_values):
        for b in itertools.product(*builder_args_values):
            t_args = dict(zip(train_args_keys, t))
            b_args = dict(zip(builder_args_keys, b))
            nn = nn_builder(**b_args)
            if cv is None:
                optimizer.train(nn, x, y, **t_args)
                loss_value = optimizer.loss.batch_mean(nn.predict_batch(x), y)
            else:
                cv_dict = cv(nn, optimizer, x, y, **t_args)
                assert 'validation_scores' in cv_dict, \
                    'The given cv function does not return a dict with key \'validation_scores\'.'
                loss_value = np.mean(cv_dict['validation_scores'])
            if loss_value < best_loss:
                best_loss = loss_value
                best_model = nn
                best_train_args = t_args
                best_builder_args = b_args

    return best_model, best_loss, best_train_args, best_builder_args


def random_search(nn_builder: Callable[[dict], NN], optimizer, x, y,
                  train_args: Dict[str, Union[rv_frozen, List[Any]]],
                  builder_args: Dict[str, Union[rv_frozen, List[Any]]],
                  iterations,
                  cv: Callable[[NN, Optimizer, object, object], Dict[str, List[float]]] = None):
    """ Perform a random search through a space of training and neural network
    topology parameters.

    An *iteration* of the random search is one of the possible combinations of
    the training and builder parameters. The first ones will be fed to
    ``optimizer.train()``, while the second ones will be fed to ``nn_builder``.

    :param nn_builder: a function that accepts a parameter ``builder_args`` and
        returns a neural network.
    :param optimizer: the optimizer to use in the random search. Its signature
        must be compatible with the keys in ``train_args``.
    :param x: a list of examples.
    :param y: a list with the target output of each example.
    :param train_args: a dictionary whose keys are:
        (1) compatible with the arguments of ``optimizer.train()``, and
        (2) associated with lists or (SciPy) random distributions that represent
        the subset of arguments to explore.
    :param builder_args: a dictionary whose keys are:
        (1) compatible with the arguments of ``nn_builder()``, and
        (2) associated with lists or (SciPy) random distributions that represent
        the subset of arguments to explore.
    :param iterations: the number of iterations of the random search.
    :param cv: an optional cross validation method that is called at each
        iteration.
    :return: a tuple with:
        (1) the best model found by the grid search,
        (2) its loss value,
        (3) arguments used to train it, and
        (4) arguments used to build it.
    """
    __check_search_args_compatibility(nn_builder, optimizer, train_args, builder_args)
    best_loss = np.inf
    best_model = None
    best_train_args = None
    best_builder_args = None

    def build_dict(args):
        return {k: v[np.random.randint(len(v))] if type(v) is list else v.rvs() for k, v in args.items()}

    print('Beginning random search with %d iterations' % iterations)

    for _ in range(iterations):
        t_args = build_dict(train_args)
        b_args = build_dict(builder_args)
        nn = nn_builder(**b_args)
        if cv is None:
            optimizer.train(nn, x, y, **t_args)
            loss_value = optimizer.loss.batch_mean(nn.predict_batch(x), y)
        else:
            cv_dict = cv(nn, optimizer, x, y, **t_args)
            assert 'validation_scores' in cv_dict, \
                'The given cv function does not return a dict with key \'validation_scores\'.'
            loss_value = np.mean(cv_dict['validation_scores'])
        if loss_value < best_loss:
            best_loss = loss_value
            best_model = nn
            best_train_args = b_args
            best_builder_args = t_args

    return best_model, best_loss, best_train_args, best_builder_args
