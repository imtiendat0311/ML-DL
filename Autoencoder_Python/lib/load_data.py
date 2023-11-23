from mnist import MNIST
import numpy as np


def load_MNIST():
    mndata = MNIST(
        '/Users/boo/Desktop/coding/Deep_Learning/MNIST')
    mndata.gz = True
    x_train, _ = mndata.load_training()
    x_test, _ = mndata.load_testing()
    x_train = np.array(x_train) / 255.0
    x_test = np.array(x_test) / 255.0
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return x_train, x_test
