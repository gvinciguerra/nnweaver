""" The :py:mod:`nnweaver.callbacks` module provides the callback classes that
will be used to execute some pre-defined or user-defined actions before, during,
or after the training of a neural network. """

import csv


class Callback(object):
    def on_epoch_end(self, epoch, nn, loss_value, metrics_values):
        raise NotImplementedError

    def on_training_begin(self, nn):
        raise NotImplementedError

    def on_training_end(self, nn):
        raise NotImplementedError


class WriteFileCallback(Callback):
    def __init__(self, filename, x_validation=None, y_validation=None, loss=None):
        """ Initialize the callback. If a data set and a loss function are
        given, it will perform a validation of the model at every epoch.

        :param filename: the output file.
        :param x_validation: the set of validation examples (optional).
        :param y_validation: the set of validation targets (optional).
        :param loss: the loss function to be evaluated in the validation step
            (optional).
        """
        assert bool(x_validation is None) == bool(y_validation is None) == bool(loss is None)
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.file = None
        self.loss = loss
        self.csv_writer = None
        self.filename = filename

    def on_training_begin(self, nn):
        """ The callback to be executed at the beginning of the training.

        :param nn: the neural network that is being trained.
        """
        self.file = open(self.filename, 'w')
        fieldnames = ['epoch', 'loss']
        if self.x_validation is not None:
            fieldnames.append('val_loss')
        self.csv_writer = csv.DictWriter(self.file, fieldnames)
        self.csv_writer.writeheader()

    def on_epoch_end(self, epoch, nn, loss_value, metrics_values):
        """ The callback to be executed at the end of every epoch.

        :param epoch: the epoch number.
        :param nn: the neural network that is being trained.
        :param loss_value: the loss value obtained in this epoch.
        :param metrics_values: the other metric values obtained in this epoch.
        """
        record = {'epoch': epoch, 'loss': loss_value}
        if self.x_validation is not None:
            y_predicted = nn.predict_batch(self.x_validation)
            validation_loss_value = self.loss(y_predicted, self.y_validation)
            record['val_loss'] = validation_loss_value
        self.csv_writer.writerow(record)

    def on_training_end(self, nn):
        """ The callback to be executed at the endo of the training.

        :param nn: the neural network that is being trained.
        """
        self.file.close()
