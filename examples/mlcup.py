import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from nnweaver import *
from optimizers import SGD

# READ DATASET
assert len(sys.argv) > 1, 'Dataset path not given'
dataset = np.genfromtxt(sys.argv[1] + '/train.csv', delimiter=',',
                        usecols=tuple(range(1, 13)))
# DO NOT TOUCH UNTIL THE END
# dataset_test = np.genfromtxt(sys.argv[1] + '/test.csv', delimiter=',',
#                              usecols=tuple(range(1, 13)))
x = dataset[:, :10]
y = dataset[:, 10:]
limit = int(len(x) * 0.8)
x_train = x[:limit]
y_train = y[:limit]
x_val = x[limit:]
y_val = y[limit:]

directory = sys.argv[1] + '/results/'
os.makedirs(directory, exist_ok=True)
training_id = len(os.listdir(directory))
sgd = SGD(MEE)
batch_size = len(x_train)


# CALLBACK TO WRITE THE NORM OF THE GRADIENT
class WriteGradientNorm(callbacks.Callback):
    def on_epoch_end(self, epoch, nn, loss_value, metrics_values):
        flattened = np.array([])
        for w, b in zip(*sgd._weights_gradient):
            flattened = np.concatenate((flattened, w.flatten(), b.flatten()))
        flattened /= batch_size
        norm = np.linalg.norm(flattened.reshape((-1, 1)))
        self.file.write('%d,%f\n' % (epoch, norm))

    def on_training_end(self, nn):
        self.file.close()

    def on_training_begin(self, nn):
        self.file = open('%s/%d_norms.csv' % (directory, training_id), 'a')
        self.file.write('epoch,gradient_norm\n')

file_path = '%s/%s.csv' % (directory, training_id)
wtf = callbacks.WriteFileCallback(file_path, x_val, y_val, loss=MEE)
plc = callbacks.PlotLearningCurve(x_val, y_val, loss=MEE, interactive=False)
wgn = WriteGradientNorm()

# DEFINE THE NN TOPOLOGY
nn = NN(10)
nn.add_layer(Layer(97, Rectifier))
nn.add_layer(Layer(19, TanH))
nn.add_layer(Layer(2, Linear))

# START THE TRAINING
regularizer = L1L2Regularizer(0.00037205665611222174, 0.0008170173137216012)
lr = learning_rate_time_based(0.010868833133798494, 5.964961555920687e-06)
sgd.train(nn, x_train, y_train,
          learning_rate=lr,
          batch_size=batch_size,
          epochs=10000,
          momentum=0.9516339410240324,
          regularizer=regularizer,
          callbacks=[plc, wtf, wgn])

fig, ax = plt.subplots()
y_predict = nn.predict_batch(x_val)
ax.scatter(y_predict[:, 0], y_predict[:, 1], c='b', marker='+', s=30, lw=.4)
ax.scatter(y_val[:, 0], y_val[:, 1], c='r', marker='+', s=30, lw=.4)
plt.show()

with open('%s/%s_predictions.csv' % (directory, training_id), 'w') as f:
    f.write('y0_predict,y1_predict,y0_val,y1_val\n')
    for y, y_val in zip(y_predict, y_val):
        f.write('%f,%f,%f,%f\n' % (y[0], y[1], y_val[0], y_val[1]))
