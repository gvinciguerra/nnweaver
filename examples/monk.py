"""This example illustrate the use of nnweaver on the MONK's problems data set.

https://archive.ics.uci.edu/ml/datasets/MONK%27s+Problems
"""

import tempfile
import urllib.request

import numpy as np

from nnweaver import *
from nnweaver.utils import accuracy, one_hot_encoding


def get_monk(monk):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/' + monk
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


x, y = get_monk('monks-3.train')
x_test, y_test = get_monk('monks-3.test')

nn = NN(17)
nn.add_layer(Layer(6, Sigmoid))
nn.add_layer(Layer(1, Sigmoid))

plc = callbacks.PlotLearningCurve(x_test, y_test, MSE, interactive=False)
sgd = SGD(MSE)
sgd.train(nn, x, y, callbacks=[plc], metrics=[accuracy],
          learning_rate=learning_rate_time_based(0.8, 0.005),
          momentum=0.5, batch_size=len(x), epochs=3000)

print('Accuracy on test %f' % accuracy(nn.predict_batch(x_test), y_test))
