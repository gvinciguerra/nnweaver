import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from nnweaver import *
from nnweaver.optimizers import ProximalBundleMethod


train_args = {'mu': 10,
              'm_L': 0.1,
              'm_R': 0.9,
              't_bar': 0.5,
              'gamma': 0.5,
              'accuracy_tolerance': 1e-4,
              'max_iterations': 100}

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

pbm = ProximalBundleMethod(MEE)
pbm.train(nn, x_train, y_train, callbacks=[plc, wtf], **train_args)

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
