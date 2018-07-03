.. hyperparameters:

Tuning Hyperparameters
======================

You can perform hyperparameter optimization with a grid search or a random search. In both cases you need to define a
function that takes some hyperparameters and builds a model::

   def nn_builder(units_1, units_2, activation_1, activation_2):
       nn = NN(1)
       nn.add_layer(Layer(units_1, activation_1))
       nn.add_layer(Layer(units_2, activation_2))
       nn.add_layer(Layer(1, Linear))
       return nn

Then, for :func:`.grid_search` you need to specify two dictionaries, one with the values for the function
defined above, the other for the arguments of the chosen optimizer::


   from nnweaver.validation import *
   from functools import partial

   builder_args = {'units_1': [5, 10, 15],
                   'units_2': [5, 10],
                   'activation_1': [Sigmoid, Rectifier],
                   'activation_2': [Sigmoid, Rectifier, Linear]}
   train_args = {'epochs': [5, 10],
                 'batch_size': [10, 15, 20],
                 'learning_rate': [0.5, 0.1]}

   sgd = SGD(MSE)
   three_fold = partial(kfold_cross_validation, k=3)
   grid_search(nn_builder, sgd, x, y, train_args, builder_args, three_fold)

If you prefer a :func:`.random_search` instead, you can specify in the dictionaries both discrete values and
distributions::

   from scipy import stats

   builder_args = {'units_1': stats.randint(5, 15),
                   'units_2': stats.randint(5, 10),
                   'activation_1': [Sigmoid, Rectifier],
                   'activation_2': [Sigmoid, Rectifier]}
   train_args = {'epochs': stats.randint(5, 10),
                 'batch_size': [10],
                 'learning_rate': stats.uniform(0.1, 0.5)}

   two_fold = partial(kfold_cross_validation, k=2)
   random_search(nn_builder, sgd, x, y, train_args, builder_args,
                 iterations=5, cv=two_fold)
