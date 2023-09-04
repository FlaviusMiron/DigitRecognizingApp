"""
Script for computing the prediction of the network and the confidence levels.
Other scripts will make use of this script.
"""

import jpg_to_list
import pickle
import numpy as np

file = open("parameters.pkl","rb")
parameters = pickle.load(file)
weights,biases = parameters


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def get_result(file_name):

    input = jpg_to_list.convert(file_name)

    a = input
    for w,b in zip(weights, biases): # FEEDFORWARDING
        a = sigmoid(np.dot(w,a) + b)

    return list(a).index(max(list(a))),a
