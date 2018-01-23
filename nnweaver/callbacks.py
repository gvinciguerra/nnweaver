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
        assert bool(x_validation is None) == bool(y_validation is None) == bool(loss is None)
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.file = None
        self.loss = loss
        self.csv_writer = None
        self.filename = filename

    def on_training_begin(self, nn):
        self.file = open(self.filename, 'w')
        fieldnames = ['epoch', 'loss']
        if self.x_validation is not None:
            fieldnames.append('val_loss')
        self.csv_writer = csv.DictWriter(self.file, fieldnames)
        self.csv_writer.writeheader()

    def on_epoch_end(self, epoch, nn, loss_value, metrics_values):
        record = {'epoch': epoch, 'loss': loss_value}
        if self.x_validation is not None:
            y_predicted = nn.predict_batch(self.x_validation)
            validation_loss_value = self.loss(y_predicted, self.y_validation)
            record['val_loss'] = validation_loss_value
        self.csv_writer.writerow(record)

    def on_training_end(self, nn):
        self.file.close()
