import numpy


def sigmoid(x):
    y = 1 / (1 + numpy.exp(-x))
    return y
