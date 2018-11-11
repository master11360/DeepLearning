from perceptron import _and, _nand, _or, _xor
from neural_network import step_function, sigmoid, init_network, forward, softmax
import numpy as np


def perceptron_test():
    inputs = (
        (0, 0), (0, 1), (1, 0), (1, 1)
    )
    for i in inputs:
        print 'AND %s = %s' % (i, _and(*i))

    for i in inputs:
        print 'NAND %s = %s' % (i, _nand(*i))

    for i in inputs:
        print 'OR %s = %s' % (i, _or(*i))

    for i in inputs:
        print 'XOR %s = %s' % (i, _xor(*i))


def neural_network_test():
    # x = np.array([-1, -2, 3])
    # print step_function(x)
    # print sigmoid(x)

    # x = np.array([-1, -2, 3])
    # print softmax(x)

    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print 'neural network output = %s' % y


def main():
    # perceptron_test()
    neural_network_test()


if __name__ == '__main__':
    main()
