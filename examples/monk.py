"""This example illustrate the use of nnweaver on the MONK's problems data set.

https://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems
"""

import tempfile
import urllib.request

import numpy as np

from nnweaver import *
from nnweaver.utils import accuracy, one_hot_encoding


def get_monk(url):
    request = urllib.request.urlopen(url)
    with tempfile.NamedTemporaryFile() as temp:
        temp.write(request.read())
        temp.flush()
        data = np.loadtxt(temp.name, dtype=np.int8, usecols=tuple(range(7)))
        x = data[:, 1:]
        y = data[:, 0]
        out = np.empty((len(x), 0))
        for i in range(x.shape[1]):
            v = one_hot_encoding(x[:, i] - 1)
            out = np.concatenate((out, v), axis=1)
        return out, y


monk1_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train'
x_train, y_train = get_monk(monk1_train)

nn = NN(17)
nn.add_layer(Layer(4, Rectifier()))
nn.add_layer(Layer(1, Sigmoid()))
SGD(MSE(), learning_rate=0.1).fit(nn, x_train, y_train, batch_size=10, epochs=100, metrics=[accuracy])

monk1_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test'
x_test, y_test = get_monk(monk1_test)
print('Accuracy on test %f' % accuracy(nn.predict_batch(x_test), y_test))
