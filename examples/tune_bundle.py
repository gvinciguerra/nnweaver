#!/usr/bin/python3

import sys
import numpy as np
import json
from scipy import stats
from nnweaver.validation import random_search, kfold_cross_validation
from nnweaver.nn import NN, Layer
from nnweaver.activations import Rectifier, TanH, Linear
from nnweaver.optimizers import ProximalBundleMethod
from nnweaver.losses import MEE
from nnweaver.regularizers import L1L2Regularizer


class LogScaleUniform(object):
    def __init__(self, low, high):
        self.log_low = np.log10(low)
        self.log_high = np.log10(high)
        self.dist = stats.uniform(0, 1)

    def rvs(self, random_state=None):
        rv = self.log_low + self.dist.rvs(random_state=random_state) * (
                self.log_high - self.log_low)
        return 10 ** rv


def nn_builder():
    nn = NN(10)
    nn.add_layer(Layer(97, Rectifier))
    nn.add_layer(Layer(19, TanH))
    nn.add_layer(Layer(2, Linear))
    return nn


train_args = {'mu': LogScaleUniform(0.1, 10),
              'm_L': stats.uniform(0.1, 0.4),
              'm_R': stats.uniform(0.5, 0.5),
              't_bar': stats.uniform(0.1, 0.9),
              'a_bar': LogScaleUniform(0.1, 100.),
              'regularizer': [L1L2Regularizer(0.00037205665611222174,
                                              0.0008170173137216012)],
              'gamma': LogScaleUniform(1e-8, 10.),
              'accuracy_tolerance': [1e-8],
              'max_iterations': [200]}

assert len(sys.argv) > 1, 'Dataset path not given'
dataset = np.genfromtxt(sys.argv[1] + '/train.csv', delimiter=',',
                        usecols=tuple(range(1, 13)))
x = dataset[:, :10]
y = dataset[:, 10:]
limit = int(len(x) * 0.8)
x_train = x[:limit]
y_train = y[:limit]
x_val = x[limit:]
y_val = y[limit:]

pbm = ProximalBundleMethod(MEE)
results = []
i = 0

while i < 10:
    try:
        _, loss, args, _ = random_search(nn_builder, pbm, x_train, y_train,
                                         train_args, {}, 1)
        element = dict()
        element["loss"] = loss
        element["args"] = args
        results.append(element)
        print("\nResults\n"
              "=======\n"
              "Loss: {}\n"
              "Parameters: {}\n\n".format(str(loss), str(args)))
    except KeyboardInterrupt:
        break
    except ValueError:
        continue
    i += 1

file_path = '%s/results.json' % sys.argv[1]

try:
    with open(file_path, 'r') as file:
        experiments = json.load(file)
except FileNotFoundError:
    experiments = []

with open(file_path, 'w+') as file:
    experiments.extend(results)
    json.dump(experiments, file, indent=3)
