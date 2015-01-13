__author__ = 'serdyuk'

import numpy as np
import cPickle as pkl
import argparse
import time


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap


def main(filename, num_feat, num_layers, **kwargs):
    train_data, valid_data, test_data = load_data(filename)

    data, labels = train_data

    model = train(data, labels, num_feat=num_feat, num_layers=num_layers)
    print test(data, labels, model)


@timing
def train(data, labels, **kwargs):
    """
    Trains model
    :param data: Training data
    :param labels: Correct labels for `data`
    :return: Trained model, list of pairs (W, b) weight matrix W and bias b
             for each layer
    """
    return None


def test(data, labels, model, **kwargs):
    """
    Computes testing error
    :param data: Testing data
    :param labels: Correct labels for `data`
    :param model: Trained model as described in function `train`
    :return: testing error
    """
    pass


def forward_prop(data, model):
    """
    Performs forward propagation and returns the last layer values
    :param data: Batch of data
    :param model: Model to use as described in `train`
    :return: List of outputs for each data point from `data`
    """
    pass


def backward_prop(data, labels, model):
    """
    Performs backprop
    :param data: Batch of data
    :param labels: Labels for batch of data
    :param model: Model as described in `train`
    :return:
    """
    pass


def load_data(filename):
    with open(filename, 'r') as fin:
        datasets = pkl.load(fin)
    return datasets


def parse_args():
    parser = argparse.ArgumentParser(description='Run MLP on MNIST')
    parser.add_argument('filename',
                        default='/data/lisa/data/mnist/mnist.pkl',
                        help='File which contains pickled dataset')
    parser.add_argument('num_feat', type=int,
                        default=784,
                        help='Number of input features')
    parser.add_argument('num_layers', type=int,
                        default=5,
                        help='Number of layers of MLP')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)
