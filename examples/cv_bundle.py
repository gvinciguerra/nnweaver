#!/usr/bin/python3

import sys
import numpy as np
import json
from nnweaver.validation import kfold_cross_validation
from nnweaver.nn import NN, Layer
from nnweaver.activations import Rectifier, TanH, Linear
from nnweaver.optimizers import ProximalBundleMethod
from nnweaver.losses import MEE
from nnweaver.regularizers import L1L2Regularizer


train_args = [
    {
        "mu": 0.6193683008171569,
        "m_L": 0.17728225397757838,
        "m_R": 0.906166719835686,
        "t_bar": 0.3039081505030413,
        "a_bar": 17.480934007619105,
        'regularizer': L1L2Regularizer(0.00037205665611222174,
                                       0.0008170173137216012),
        "gamma": 1.0259059587162216e-07,
        "accuracy_tolerance": 1e-08,
        "max_iterations": 200
    }, {
        "mu": 0.4173321474040356,
        "m_L": 0.07885015992295061,
        "m_R": 0.9776582163043794,
        "t_bar": 0.5597473855993359,
        "a_bar": 68.98979835558536,
        'regularizer': L1L2Regularizer(0.00037205665611222174,
                                       0.0008170173137216012),
        "gamma": 0.000360105914385689,
        "accuracy_tolerance": 1e-08,
        "max_iterations": 200
    }, {
        "mu": 0.685620461876153,
        "m_L": 0.13275083615739208,
        "m_R": 0.8686328845272789,
        "t_bar": 0.678970490404228,
        "a_bar": 10.088896580382626,
        'regularizer': L1L2Regularizer(0.00037205665611222174,
                                       0.0008170173137216012),
        "gamma": 0.001536084428942603,
        "accuracy_tolerance": 1e-08,
        "max_iterations": 200
    }, {
        "mu": 0.6846958549814486,
        "m_L": 0.3589779459957336,
        "m_R": 0.7693745903055,
        "t_bar": 0.5152351571428506,
        "a_bar": 9.69604585203311,
        'regularizer': L1L2Regularizer(0.00037205665611222174,
                                       0.0008170173137216012),
        "gamma": 0.0001520582501448714,
        "accuracy_tolerance": 1e-08,
        "max_iterations": 200
    }, {
        "mu": 1.5390579125627797,
        "m_L": 0.24819754822019982,
        "m_R": 0.7133994019383036,
        "t_bar": 0.10966409889986806,
        "a_bar": 49.59510824209799,
        'regularizer': L1L2Regularizer(0.00037205665611222174,
                                       0.0008170173137216012),
        "gamma": 0.022270333763981298,
        "accuracy_tolerance": 1e-08,
        "max_iterations": 200
    }
]

assert len(sys.argv) > 1, 'Dataset path not given'
dataset = np.genfromtxt(sys.argv[1] + '/train.csv', delimiter=',',
                        usecols=tuple(range(1, 13)))
x = dataset[:, :10]
y = dataset[:, 10:]
results = []

for args in train_args:
    nn = NN(10)
    nn.add_layer(Layer(97, Rectifier))
    nn.add_layer(Layer(19, TanH))
    nn.add_layer(Layer(2, Linear))

    pbm = ProximalBundleMethod(MEE)
    result = dict()
    result['results'] = kfold_cross_validation(nn, pbm, x, y, k=3, **args)
    result['args'] = args
    result['args']['regularizer'] = {"l1": args['regularizer'].l1,
                                     "l2": args['regularizer'].l2}
    results.append(result)

file_path = '%s/cv_results.json' % sys.argv[1]

with open(file_path, 'w+') as file:
    json.dump(results, file, indent=3)
