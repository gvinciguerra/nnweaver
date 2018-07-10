import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from nnweaver import *
from nnweaver.optimizers import ProximalBundleMethod

regularizer = L1L2Regularizer(0.00037205665611222174, 0.0008170173137216012)

train_args = {
    "mu": 1.226627590424536,
    "m_L": 0.21547347050343577,
    "m_R": 0.549492896940917,
    "t_bar": 0.28027206476236083,
    "a_bar": 4.829740742364858,
    "regularizer": regularizer,
    "gamma": 0.0053088012268946985,
    "accuracy_tolerance": 1e-08,
    "max_iterations": 200
}


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

nn = NN(10)
nn.add_layer(Layer(97, Rectifier))
nn.add_layer(Layer(19, TanH))
nn.add_layer(Layer(2, Linear))

directory = sys.argv[1] + '/results/'
os.makedirs(directory, exist_ok=True)
file_path = '%s/%s.csv' % (directory, len(os.listdir(directory)))
wtf = callbacks.WriteFileCallback(file_path, x_val, y_val, loss=MEE)
plc = callbacks.PlotLearningCurve(x_val, y_val, loss=MEE, interactive=True)

pbm = ProximalBundleMethod(MEE)
pbm.train(nn, x_train, y_train, callbacks=[plc, wtf], **train_args)

with open(file_path, 'a') as f:
    description = '# %s\n# %s\n' % (str(nn), train_args)
    f.write(description)

fig, ax = plt.subplots()
y_predict = nn.predict_batch(x_val)
ax.scatter(y_predict[:, 0], y_predict[:, 1],
           c='b', marker='+', s=30, linewidth=.4)
ax.scatter(y_val[:, 0], y_val[:, 1], c='r', marker='+', s=30, linewidth=.4)
plt.show()

with open('%s/bundle_predictions.csv' % (directory), 'w') as f:
    f.write('y0_predict,y1_predict,y0_val,y1_val\n')
    for y, y_val in zip(y_predict, y_val):
        f.write('%f,%f,%f,%f\n' % (y[0], y[1], y_val[0], y_val[1]))

# DO NOT TOUCH UNTIL THE END
# dataset_test = np.genfromtxt(sys.argv[1] + '/test.csv', delimiter=',', usecols=tuple(range(1, 13)))
