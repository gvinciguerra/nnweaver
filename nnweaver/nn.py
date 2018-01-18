class NN(object):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.layers = []

    def add_layer(self, layer):
        if len(self.layers) == 0:
            layer.build_weights(self.input_dim)
        else:
            preceding_units = self.layers[-1].units
            layer.build_weights(preceding_units)

        self.layers.append(layer)

    def predict(self, x):
        xi = x
        for l in self.layers:
            xi = l(xi)

        return xi

    def train(self, optimizer):
        optimizer(self)
