class NN(object):
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def add_layer(self, layer):
        raise NotImplementedError
