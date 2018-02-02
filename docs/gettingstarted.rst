Getting Started
===============

Specify a Neural Network Topology
---------------------------------

You can create a feedforward neural network specifying the number of inputs as the argument of :class:`.NN`, and the
number of outputs by adding a :class:`.Layer`::

   from nnweaver import *
   nn = NN(3)
   nn.add_layer(Layer(5, Linear))

You can always add more layers, specify an activation function and a weights initializer, as the following lines of code
show::

   nn.add_layer(Layer(7, Sigmoid))
   nn.add_layer(Layer(6, Rectifier, uniform(0, 0.05)))
   nn.add_layer(Layer(42, TanH, glorot_uniform()))

See :mod:`.activations` for the list of available activation functions.

Train the Neural Network
------------------------

Now, choose a :class:`.Loss` function, pass it to an :class:`.Optimizer` (like the stochastic gradient
descent) and start the training::

   sgd = SGD(MSE)
   sgd.train(nn, x, y, learning_rate=0.3)

There are other arguments to pass to the :py:meth:`.SGD.train` method, for example::

   sgd.train(nn, x_train, y_train,
             learning_rate_time_based(0.25, 0.001),
             batch_size=10, epochs=100, momentum=0.85)

Also, you may want to control the model complexity. :meth:`.SGD.train` has a ``regularizer`` argument, that
accepts an instance of the :class:`.L1L2Regularizer` class.

Validate a Model
----------------

The :mod:`.validation` module has some facilities to assess a model and select its hyperparameters. These lines of code,
for example, perform a 3-fold cross validation::

   from nnweaver.validation import *
   results = kfold_cross_validation(nn, sgd, x, y, k=3, learning_rate=0.5, batch_size=5, epochs=100)
   print(results)

Refer to :doc:`hyperparameters` for more complex examples.


Other (Cool) Stuff
------------------

With the :mod:`.callbacks` mechanism you can get information on the training procedure. There are two callbacks in
NNWeaver to write to a file and to plot the loss for each epoch on a training and a validation set::

   from nnweaver.callbacks import *

   write = callbacks.WriteFileCallback('out.csv')
   plot = callbacks.PlotLearningCurve(x_val, y_val, MSE, interactive=True)

   sgd = SGD(MSE)
   sgd.train(nn, x, y, learning_rate=0.3, batch_size=25, epochs=170,
             callbacks=[write, plot])

You can implement your own callback extending the class :class:`.callbacks.Callback`.