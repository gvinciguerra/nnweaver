import gzip
import struct
import urllib.request

import numpy as np

from nnweaver import *
from nnweaver.utils import accuracy, one_hot_encoding


def get_mnist(train):
    prefix = 'train' if train else 't10k'
    headers = {"Accept-Encoding": "gzip"}
    url = 'http://yann.lecun.com/exdb/mnist/'
    images_file = prefix + '-images-idx3-ubyte.gz'
    labels_file = prefix + '-labels-idx1-ubyte.gz'
    req_1 = urllib.request.Request(url + images_file, headers=headers)
    req_2 = urllib.request.Request(url + labels_file, headers=headers)
    images_req = urllib.request.urlopen(req_1)
    labels_req = urllib.request.urlopen(req_2)

    with gzip.open(labels_req) as file:
        magic, num = struct.unpack(">II", file.read(8))
        buf = file.read(num)
        labels = np.frombuffer(buf, dtype=np.uint8)
        y = one_hot_encoding(labels)

    with gzip.open(images_req) as file:
        magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
        buf = file.read(num * rows * cols)
        x = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)

    return x / 255, y, labels


x_train, y_train, _ = get_mnist(True)
x_test, y_test, test_labels = get_mnist(False)

nn = NN(28 * 28)
nn.add_layer(Layer(30, Sigmoid()))
nn.add_layer(Layer(10, Sigmoid()))
SGD(MSE()).train(nn, x_train, y_train,
                 learning_rate=0.5, batch_size=50, epochs=10,
                 metrics=[accuracy],
                 regularizer=L1L2Regularizer(l1=0.00, l2=0.005))

y_predicted = nn.predict_batch(x_test)
print('Accuracy on test %f' % accuracy(y_predicted, y_test))


# Uncomment this to display errors
# import matplotlib.pyplot as plt
# predictions = np.argmax(y_predicted, 1)
# errors = (predictions != test_labels)
# for img, n_error, n in zip(x_test[errors], predictions[errors], test_labels[errors]):
#     plt.figure()
#     print('Output %d. Was %d.' % (n_error, n))
#     plt.imshow(img.reshape((28, 28)), cmap='gray')
#     plt.show()
#     input()
