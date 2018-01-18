import numpy as np
import tqdm
from math import ceil


class Optimizer(object):
    def __init__(self, loss):  # TODO: Regularization
        self.loss = loss


class GradientBasedOptimizer(Optimizer):
    def forward(self, nn, x):
        """ Propagate an input signal through the network.

        :param nn: the neural network.
        :param x: the input.
        :return: two lists of length len(nn.layers) that contains,
            respectively, the input sum and the output (i.e. the result of
            applying the activation function on the input sum) at each layer.
        """
        outputs = []
        inputs = []
        xi = x

        for l in nn.layers:
            inputs.append(xi)
            xi = l(xi)
            outputs.append(xi)

        return inputs, outputs

    def backward(self, nn, y, inputs, outputs):
        """ Propagate an error backward through the network.

        :param nn: the neural network.
        :param y: the target output.
        :param inputs: a list of the input sums at each layer.
        :param outputs: a list of the activated outputs at each layer.
        :return: a list of the gradients of the loss function at each layer.
        """
        errors = []  # list of size len(nn.layers)
        layers_reversed = range(len(nn.layers) - 1, -1, -1)

        for l in layers_reversed:
            if l == len(nn.layers) - 1:
                grad_out = self.loss.gradient(outputs[l], y)
            else:
                grad_out = nn.layers[l + 1].weights.T.dot(errors[0])
            chain = nn.layers[l].activation.gradient(inputs[l])
            grad_pre_act = np.multiply(grad_out, chain)
            errors.insert(0, grad_pre_act)

        return errors


class SGD(GradientBasedOptimizer):
    def __init__(self, loss, learning_rate=0.05):
        super().__init__(loss)
        self.learning_rate = learning_rate

    @staticmethod
    def batch_ranges(X, batch_size):
        size = len(X)
        num_batches = int(ceil(size / float(batch_size)))
        return [(i * batch_size, min(size, (i + 1) * batch_size))
                for i in range(num_batches)]

    def fit(self, nn, X, Y, batch_size=1, epochs=1):
        assert batch_size <= len(X)

        ranges = SGD.batch_ranges(X, batch_size)
        step = self.learning_rate / batch_size
        for epoch in range(0, epochs):
            bar = tqdm.tqdm(ranges, desc="Epoch %3d" % epoch)

            for low, high in ranges:
                # Estimate gradient
                tot_errors = [0 for _ in range(len(nn.layers))]
                tot_errors_bias = [0 for _ in range(len(nn.layers))]
                for i, o in zip(X[low:high], Y[low:high]):
                    inputs, outputs = self.forward(nn, i)
                    errors = self.backward(nn, o, inputs, outputs)
                    for l in range(len(nn.layers)):
                        tot_errors_bias[l] += errors[l]
                        tot_errors[l] += errors[l].dot(inputs[l].T)
                tot_errors = [e / batch_size for e in tot_errors]
                tot_errors_bias = [e / batch_size for e in tot_errors_bias]

                # Update weights
                for (lay, grad, grad_b) in zip(nn.layers, tot_errors, tot_errors_bias):
                    lay.weights -= step * grad
                    lay.bias -= step * grad_b

                bar.set_postfix(Loss=np.mean(tot_errors[-1]))
                bar.update(1)

            bar.close()
