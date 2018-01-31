import tempfile
import urllib.request

import numpy as np

from nnweaver import *


def get_winequality():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    request = urllib.request.urlopen(url)
    with tempfile.NamedTemporaryFile() as temp:
        temp.write(request.read())
        temp.flush()
        csv = np.loadtxt(temp.name, dtype=str, delimiter=';')
        features = csv[0]
        data = csv[1:].astype(float)
        x = data[:, :-1]
        y = data[:, -1]
        x /= x.max(axis=0)
        y /= y.max(axis=0)
        return x, y, features


x, y, features = get_winequality()
limit = int(0.8 * len(x))
x_train, y_train = x[:limit], y[:limit]
x_test, y_test = x[limit:], y[limit:]

nn = NN(11)
nn.add_layer(Layer(1, Sigmoid))

sgd = SGD(MSE)
sgd.train(nn, x_train, y_train, epochs=5, batch_size=20, learning_rate=1)

for f, w in zip(features, nn.layers[0].weights.reshape(-1)):
    print('%s = %f' % (f, w))
print('Loss on test %f' % MSE.batch_mean(nn.predict_batch(x_test), y_test))
