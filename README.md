<p align="center">
  <img src="https://gvinciguerra.github.io/nnweaver/_static/logo.svg">
</p>

# NNWeaver #

[![Coverage Status](https://coveralls.io/repos/github/gvinciguerra/nnweaver/badge.svg?branch=master)](https://coveralls.io/github/gvinciguerra/nnweaver?branch=master)
[![Build Status](https://travis-ci.org/gvinciguerra/nnweaver.svg?branch=master)](https://travis-ci.org/gvinciguerra/nnweaver)

NNWeaver is a *tiny* Python library to create and train feedforward neural networks. We developed this library as a project for a Machine Learning course.

Some of its features are:

  1. Simple API, easy to learn.
  2. Validation functions included.
  3. Lightweight and with few dependencies.
  4. Live loss/epoch curve display.

## Installation ##

You can install NNWeaver from the GitHub source with the following commands:

    git clone https://github.com/gvinciguerra/nnweaver.git
    cd nnweaver
    python3 setup.py install

You can also run the test suite with `python3 setup.py test`.

## Getting started ##

### Specify a Neural Network Topology ###

You can create a feedforward neural network specifying the number of inputs as the argument of [`NN`](https://gvinciguerra.github.io/nnweaver/nnweaver.html#nnweaver.nn.NN), and the number of outputs by adding a [`Layer`](https://gvinciguerra.github.io/nnweaver/nnweaver.html#nnweaver.nn.Layer):

    from nnweaver import *
    nn = NN(3)
    nn.add_layer(Layer(5, Linear))

You can always add more layers, specify an activation function and a weights initializer, as the following lines of code show:

    nn.add_layer(Layer(7, Sigmoid))
    nn.add_layer(Layer(6, Rectifier, uniform(0, 0.05)))
    nn.add_layer(Layer(42, TanH, glorot_uniform()))

See [`activations`](https://gvinciguerra.github.io/nnweaver/nnweaver.html#module-nnweaver.activations) for the list of available activation functions.

### Train the Neural Network ###

Now, choose a [`Loss`](https://gvinciguerra.github.io/nnweaver/nnweaver.html#nnweaver.losses.Loss) function, pass it to an [`Optimizer`](https://gvinciguerra.github.io/nnweaver/nnweaver.html#nnweaver.optimizers.Optimizer) (like the stochastic gradient descent) and start the training:

    sgd = SGD(MSE)
    sgd.train(nn, x, y, learning_rate=0.3)

There are other arguments to pass to the [`SGD.train()`](https://gvinciguerra.github.io/nnweaver/nnweaver.html#nnweaver.optimizers.SGD.train) method, for example:

    sgd.train(nn, x_train, y_train,
              learning_rate_time_based(0.25, 0.001),
              batch_size=10, epochs=100, momentum=0.85)

Also, you may want to control the model complexity. [`SGD.train()`](https://gvinciguerra.github.io/nnweaver/nnweaver.html#nnweaver.optimizers.SGD.train) has a `regularizer` argument, that accepts an instance of the [`L1L2Regularizer`](https://gvinciguerra.github.io/nnweaver/nnweaver.html#nnweaver.regularizers.L1L2Regularizer) class.

### A very, very simple example ###

<img src="https://github.com/gvinciguerra/nnweaver/blob/gh-pages/_images/nnweaver.gif?raw=true" width="550" />

## Documentation ##

For more information, tutorials, and API reference, please visit [NNweaver's online documentation](https://gvinciguerra.github.io/nnweaver/index.html) or build your own offline copy executing `python3 setup.py docs`.

## License ##

This project is licensed under the terms of the MIT License.
