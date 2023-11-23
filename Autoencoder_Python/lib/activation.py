import numpy as np


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (-1 + 2 / (1 + np.exp(-2*x)))


def identity(x):
    return x

# derivative of activation functions


def d_relu(x):
    return np.where(x > 0, 1, 0)


def d_sigmoid(x):
    return x * (1-x)


def d_tanh(x):
    return 1 - x**2


def d_identity(x):
    return 1

activation = {
        "relu": relu,
        "sigmoid": sigmoid,
        "tanh": tanh,
        "identity": identity
}
derivative = {
        "relu": d_relu,
        "sigmoid": d_sigmoid,
        "tanh": d_tanh,
        "identity": d_identity
}