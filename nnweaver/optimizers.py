class Optimizer(object):
    def __call__(self, nn, X, Y):
        raise NotImplementedError


class GradientBasedOptimizer(Optimizer):
    def __call__(self, nn, X, Y):
        raise NotImplementedError

    def __init__(self, loss):  # TODO: Regularization
        self.loss = loss

    def backward(self):
        pass


class SGD(GradientBasedOptimizer):
    def __call__(self, nn, X, Y):
        pass
