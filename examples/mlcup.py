import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from nnweaver import *
from optimizers import SGD

train_args = {'learning_rate': 0.010868833133798494,
              'decay': 5.964961555920687e-06,
              'l1': 0.00037205665611222174,
              'l2': 0.0008170173137216012,
              'batch_size': 5,
              'epochs': 5,
              'momentum': 0.9516339410240324}

assert len(sys.argv) > 1, 'Dataset path not given'
dataset = np.genfromtxt(sys.argv[1] + '/train.csv', delimiter=',', usecols=tuple(range(1, 13)))
x = dataset[:, :10]
y = dataset[:, 10:]
limit = int(len(x) * 0.8)
x_train = x[:limit]
y_train = y[:limit]
x_val = x[limit:]
y_val = y[limit:]

nn = NN(10)
nn.add_layer(Layer(97, Rectifier))
nn.add_layer(Layer(19, TanH))
nn.add_layer(Layer(2, Linear))

directory = sys.argv[1] + '/results/'
os.makedirs(directory, exist_ok=True)
file_path = '%s/%s.csv' % (directory, len(os.listdir(directory)))
wtf = callbacks.WriteFileCallback(file_path, x_val, y_val, loss=MEE)
plc = callbacks.PlotLearningCurve(x_val, y_val, loss=MEE, interactive=False)

extra = dict()
if 'l1' in train_args or 'l2' in train_args:
    extra['regularizer'] = L1L2Regularizer(train_args['l1'], train_args['l2'])
extra['learning_rate'] = learning_rate_time_based(train_args['learning_rate'], train_args['decay'])
sgd = SGD(MEE)
sgd.train(nn, x_train, y_train, callbacks=[plc, wtf],
          batch_size=train_args['batch_size'], epochs=train_args['epochs'], momentum=train_args['momentum'], **extra)

with open(file_path, 'a') as f:
    description = '# %s\n# %s\n' % (str(nn), train_args)
    f.write(description)

fig, ax = plt.subplots()
y_predict = nn.predict_batch(x_val)
ax.scatter(y_predict[:, 0], y_predict[:, 1], c='b', marker='+', s=30, linewidth=.4)
ax.scatter(y_val[:, 0], y_val[:, 1], c='r', marker='+', s=30, linewidth=.4)
plt.show()

# DO NOT TOUCH UNTIL THE END
# dataset_test = np.genfromtxt(sys.argv[1] + '/test.csv', delimiter=',', usecols=tuple(range(1, 13)))
