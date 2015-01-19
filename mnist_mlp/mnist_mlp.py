__author__ = 'serdyuk'

import numpy as np
from itertools import tee, izip
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


def sigma(x):
    return 1. / (1. + np.exp(-x))


def sigma_grad(y):
    return y * (1. - y)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def softmax_grad(y):
    m, n = y.shape
    delta = np.zeros((m, n, n))
    delta[:, xrange(n), xrange(n)] = 1.
    return y[:, :, None] * (delta - y[:, None, :])


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


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
    pass


def test(data, labels, model, **kwargs):
    """
    Computes testing error
    :param data: Testing data
    :param labels: Correct labels for `data`
    :param model: Trained model as described in function `train`
    :return: testing error
    """
    pass


def init_model(num_feat, layer_sizes, num_out, seed=1):
    model = []
    rng = np.random.RandomState(seed)
    layer_sizes = [num_feat] + layer_sizes + [num_out]
    for size in pairwise(layer_sizes):
        W = rng.normal(scale=0.1, size=size).T
        b = rng.normal(scale=0.1, size=(size[1], 1))
        model += [(W, b)]
    return model


def forward_prop(batch, model):
    """
    Performs forward propagation and returns the last layer values
    :param batch: Batch of data
    :param model: Model to use as described in `train`
    :return: List of outputs for each data point from `data` and values
             obtained
    """
    hidden = batch
    values = []
    for W, b in model[:-1]:
        value = hidden.dot(W.T) + b.T
        hidden = sigma(value)
        values += [value.copy()]
    V, c = model[-1]
    val = hidden.dot(V.T) + c.T
    outputs = softmax(val)
    values += [val.copy()]
    return outputs, values


def backprop_step(dEdY, W, gd, Y, X):
    deltas = (gd(Y) * dEdY)
    dEdX = deltas.dot(W)
    dEdW = deltas.T.dot(X) / deltas.shape[0]
    dEdb = deltas.mean(axis=0)
    return dEdX, dEdW, dEdb


def compute_loss(labels, outputs):
    return (-labels * np.log(outputs)).sum(axis=1).mean()


def backward_prop(data, labels, model, values, outputs):
    """
    Performs backprop
    :param data: Batch of data (batch_index, data_index)
    :param labels: Labels for batch of data
    :param model: Model as described in `train`
    :param values: Hidden variable values from forward propagation
    :param outputs: Outputs from forward propagation
    :return: All derivatives
    """
    # matrix (batch_index, output_index)
    err = ((-labels * (1. / outputs))[:, :, None] *
           softmax_grad(outputs)).sum(axis=1)
    err, w_diff, b_diff = backprop_step(err, model[-1][0],
                                        lambda x: np.ones_like(x),
                                        values[-1], values[-2])
    diffs = [(w_diff.copy(), b_diff.copy())]
    values = [data] + values
    for i, (W, b) in enumerate(reversed(model[:-1])):
        err, w_diff, b_diff = backprop_step(err, W, sigma_grad,
                                            values[-i-2], values[-i-3])
        diffs += [(w_diff.copy(), b_diff.copy())]

    return [x for x in reversed(diffs)]


def one_hot(data, out_size):
    out = np.zeros((len(data), out_size))
    out[(xrange(len(data)), data)] = 1.
    return out


def load_data(filename, out_size):
    datasets = []
    with open(filename, 'r') as fin:
        data = pkl.load(fin)
    for dataset in data:
        features, labels = dataset
        datasets += [[features, one_hot(labels, out_size)]]

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
    parser.add_argument('out_size', type=int,
                        default=10,
                        help='Size of output layer')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)
