import copy

import numpy as np
import time


def splits_generator(x, y, groups):
    assert len(x) == len(y)
    assert sum(groups) == len(x)

    begin = 0
    for g in groups:
        end = begin + g
        yield x[begin:end], y[begin:end], \
              np.concatenate((x[:begin], x[end:])), \
              np.concatenate((y[:begin], y[end:]))
        begin += g


def clone_nn(nn):
    result = copy.deepcopy(nn)
    result.reset()
    return result


def kfold_cross_validation(nn, optimizer, x, y, k=3, **train_args):
    split_size = int(len(x) / k)
    groups = [split_size for _ in range(k - 1)] + [len(x) - split_size * (k - 1)]
    best_loss = np.inf
    best_model = None
    fold = 1

    for split_x, split_y, split_complement_x, split_complement_y in splits_generator(x, y, groups):
        nn_i = clone_nn(nn)
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
