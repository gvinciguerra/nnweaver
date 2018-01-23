import tempfile

import numpy as np

from nnweaver import *
from nnweaver.callbacks import *


def test_write_file_callback():
    x = np.arange(-1, 1, 0.05)
    y = np.arange(-1, 1, 0.05)
    limit = int(len(x) * 0.8)

    nn = NN(1)
    nn.add_layer(Layer(1, Linear()))
    sgd = SGD(MSE())

    with tempfile.NamedTemporaryFile() as tmp:
        epochs = 3
        callback = WriteFileCallback(tmp.name, x[limit:], y[limit:], MSE())
        sgd.train(nn, x[:limit], y[:limit], 0.1, 5, epochs, callbacks=[callback])
        tmp.file.seek(0)
        assert sum(1 for _ in tmp.file) == epochs + 1
