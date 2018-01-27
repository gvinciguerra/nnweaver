""" The :py:mod:`nnweaver.callbacks` module provides the callback classes that
will be used to execute some pre-defined or user-defined actions before, during,
or after the training of a neural network. """

import csv
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt


class Callback(ABC):
    """ Abstract base class for classes that provide methods that can be
    called during the :py:meth:`.Optimizer.train` execution.
    """

    @abstractmethod
    def on_epoch_end(self, epoch, nn, loss_value, metrics_values):
        pass

    @abstractmethod
    def on_training_begin(self, nn):
        pass

    @abstractmethod
    def on_training_end(self, nn):
        pass


class WriteFileCallback(Callback):
    def __init__(self, filename, x_validation=None, y_validation=None, loss=None, metrics=None):
        """ Create a callback that, at each epoch, writes the loss found by a
        training algorithm to a CSV file. If a data set and a loss function are
        given, it will also write to the file the validation results.

        :param filename: the output file.
        :param x_validation: the set of validation examples.
        :param y_validation: the set of validation targets.
        :param loss: the loss function to be evaluated in the validation step.
        :param metrics: a list of metrics to be evaluated in the validation step.
        """
        assert bool(x_validation is None) == bool(y_validation is None) == bool(loss is None)
        assert metrics is None or x_validation is not None
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.file = None
        self.loss = loss
        self.csv_writer = None
        self.metrics = [] if metrics is None else metrics
        self.filename = filename

    def on_training_begin(self, nn):
        """ The callback to be executed at the beginning of the training.

        :param nn: the neural network that is being trained.
        """
        self.file = open(self.filename, 'w')
        self.csv_writer = None

    def on_epoch_end(self, epoch, nn, loss_value, metrics_values):
        """ The callback to be executed at the end of every epoch.

        :param epoch: the epoch number.
        :param nn: the neural network that is being trained.
        :param loss_value: the loss value obtained in this epoch.
        :param metrics_values: the other metric values obtained in this epoch.
        """
        if self.csv_writer is None:
            fieldnames = ['epoch', 'loss']
            if self.x_validation is not None:
                fieldnames.append('val_loss')
            fieldnames.extend(['train_' + m for m in metrics_values.keys()])
            fieldnames.extend(['val_' + m.__name__ for m in self.metrics])
            self.csv_writer = csv.DictWriter(self.file, fieldnames)
            self.csv_writer.writeheader()

        record = {'epoch': epoch, 'loss': loss_value}
        for k, v in metrics_values.items():
            record['train_' + k] = v
        if self.x_validation is not None:
            y_predicted = nn.predict_batch(self.x_validation)
            validation_loss_value = self.loss.batch_mean(y_predicted, self.y_validation)
            record['val_loss'] = validation_loss_value
            for m in self.metrics:
                record['val_' + m.__name__] = m(y_predicted, self.y_validation)
        self.csv_writer.writerow(record)

    def on_training_end(self, nn):
        """ The callback to be executed at the end of the training.

        :param nn: the neural network that is being trained.
        """
        self.file.close()


class PlotLearningCurve(Callback):
    def __init__(self, x_validation=None, y_validation=None, loss=None, interactive=True, max_epochs=None):
        """ Create a callback that plot the learning curve of a model during and
        after its training phase. If a data set and a loss function are given,
        it will also plot the validation results.

        :param x_validation: the set of validation examples.
        :param y_validation: the set of validation targets.
        :param loss: the loss function to be evaluated in the validation step.
        :param interactive: if ``True``, show the learning curve interactively
            during the training of the model, otherwise only after the training.
            Notice that the former may cause a loss of training performance.
        :param max_epochs: the expected number of epochs. If present, this
            parameter prevents the plot from rescaling during the training (if
            ``interactive`` is ``False``, this parameter is ignored).
        """
        assert bool(x_validation is None) == bool(y_validation is None) == bool(loss is None)
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.loss = loss
        self.epochs = []
        self.losses = []
        self.val_losses = []
        self.fig, self.ax = plt.subplots()
        self.interactive = interactive
        self.max_epochs = max_epochs

    def on_training_begin(self, nn):
        """ The callback to be executed at the beginning of the training.

        :param nn: the neural network that is being trained.
        """
        if self.interactive:
            plt.ion()
            plt.show()

    def on_epoch_end(self, epoch, nn, loss_value, metrics_values=None):
        """ The callback to be executed at the end of every epoch.

        :param epoch: the epoch number.
        :param nn: the neural network that is being trained.
        :param loss_value: the loss value obtained in this epoch.
        :param metrics_values: ignored.
        """
        self.epochs.append(epoch)
        self.losses.append(loss_value)
        if self.x_validation is not None:
            y_predicted = nn.predict_batch(self.x_validation)
            validation_loss_value = self.loss.batch_mean(y_predicted, self.y_validation)
            self.val_losses.append(validation_loss_value)

        if self.interactive:
            self.ax.cla()
            self.ax.plot(self.epochs, self.losses, 'b', label='Training')
            if self.x_validation is not None:
                self.ax.plot(self.epochs, self.val_losses, 'r', label='Validation')
                self.ax.legend()
            if self.max_epochs:
                self.ax.set_xlim(0, self.max_epochs)
            self.ax.set_ylim(0)
            self.ax.set_title('Learning Curve')
            self.ax.set_xlabel('Epoch')
            self.ax.set_ylabel('Loss')
            self.fig.canvas.draw()
            plt.pause(1e-16)

    def on_training_end(self, nn):
        """ The callback to be executed at the end of the training.

        :param nn: the neural network that is being trained.
        """
        if self.interactive:
            plt.ioff()
        else:
            self.ax.set_title('Learning Curve')
            self.ax.set_xlabel('Epoch')
            self.ax.set_ylabel('Loss')
            self.ax.plot(self.epochs, self.losses, 'b')
            if self.x_validation is not None:
                self.ax.plot(self.epochs, self.val_losses, 'r')
        plt.show(block=True)
