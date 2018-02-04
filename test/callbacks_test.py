import tempfile
from threading import Thread

import numpy as np

from nnweaver import *
from nnweaver.callbacks import *
from nnweaver.optimizers import Optimizer


def test_write_file_callback():
    x = np.arange(-1, 1, 0.05)
    y = np.arange(-1, 1, 0.05)
    limit = int(len(x) * 0.8)

    nn = NN(1)
    nn.add_layer(Layer(1, Linear))
    sgd = SGD(MSE())

    with tempfile.NamedTemporaryFile() as tmp:
        epochs = 3
        callback = WriteFileCallback(tmp.name, x[limit:], y[limit:], MSE)
        sgd.train(nn, x[:limit], y[:limit], 0.1, 5, epochs, callbacks=[callback])
        tmp.file.seek(0)
        assert sum(1 for _ in tmp.file) == epochs + 1


def test_plot_learning_curve():
    samples = 50
    a = np.random.uniform(0, 2 * np.pi, samples * 2)
    r = np.append(np.random.uniform(0, 10, samples), np.random.uniform(20, 30, samples))
    x = np.matrix([np.multiply(r, np.sin(a)), np.multiply(r, np.cos(a))]).T
    y = np.append(np.ones(samples), np.zeros(samples))
    x, y = Optimizer.shuffle(x, y)
    limit = int(len(x) * 0.8)

    nn = NN(2)
    nn.add_layer(Layer(4, Rectifier))
    nn.add_layer(Layer(1, Sigmoid))
    sgd = SGD(MSE)
    sgd.seed = 42
    epochs = 3
    callback = callbacks.PlotLearningCurve(x[limit:], y[limit:], loss=MSE, max_epochs=epochs, blocking=False)
    sgd.train(nn, x[:limit], y[:limit], 0.1, 5, epochs, callbacks=[callback])
    plt.close('all')
