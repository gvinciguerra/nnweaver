class Activation(object):
    def __call__(self, x):
        raise NotImplementedError

    def gradient(self, x):
        raise NotImplementedError


class Linear(Activation):
    def __call__(self, x):
        return x

    def gradient(self, x):
        return 1


class Rectifier(Activation):
    def __call__(self, x):
        return max(0, x)

    def gradient(self, x):
        return 1 if x > 0 else 0
