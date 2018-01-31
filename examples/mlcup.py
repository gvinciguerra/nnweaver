import numpy as np
import os

from nnweaver import *

dataset = np.genfromtxt('/Users/giorgio/Desktop/training.csv', delimiter=',', usecols=tuple(range(1, 13)))

x = dataset[:, :10]
y = dataset[:, 10:]
limit = int(len(x) * 0.8)
x_train = x[:limit]
y_train = y[:limit]
x_val = x[limit:]
y_val = y[limit:]

nn = NN(10)
nn.add_layer(Layer(114, TanH))
nn.add_layer(Layer(2, Linear))


directory = '/Users/giorgio/Desktop/cup-experiments'
plc = callbacks.PlotLearningCurve(x_val, y_val, loss=MEE, interactive=False)
wtf = callbacks.WriteFileCallback(directory + '/%s.csv' % len(os.listdir(directory)), x_val, y_val, loss=MEE)

sgd = SGD(MEE)
sgd.train(nn, x_train, y_train,
          learning_rate=learning_rate_time_based(0.3154566664975894, 0.017680596168913376),
          regularizer=L1L2Regularizer(0.000227309882866499, 5.178824211523914e-05),
          batch_size=206, epochs=1206, momentum=0.9135299496450809, callbacks=[wtf, plc])


# DO NOT TOUCH UNTIL THE END
# dataset_test = np.genfromtxt('/Users/giorgio/Desktop/test.csv', delimiter=',', usecols=tuple(range(1, 13)))