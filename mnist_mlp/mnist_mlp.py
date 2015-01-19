__author__ = 'serdyuk'

from itertools import tee, izip
import cPickle as pkl
import argparse
import time
import gzip

import numpy as np


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
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


def main(filename, num_feat, num_layers, out_size, batch_size, n_epochs,
         **kwargs):
    train_data, valid_data, test_data = load_data(filename, out_size,
                                                  batch_size)

    train_data, train_labels = train_data
    valid_data, valid_labels = valid_data
    test_data, test_labels = test_data

    rng = np.random.RandomState(1)
    model = train(train_data, train_labels, valid_data, valid_labels,
                  n_epochs, 1e-2, rng, num_feat=num_feat, num_layers=num_layers)
    with open('model.pkl', 'w') as fout:
        pkl.dump(model, fout)

    loss, misclass = test(test_data, test_labels, model)
    print '.. testing error', loss, 'misclassification', misclass


@timing
def train(data, labels, valid_data, valid_labels, n_epochs,
          learn_rate, rng, **kwargs):
    """
    Trains model
    :param data: Training data
    :param labels: Correct labels for `data`
    :return: Trained model, list of pairs (W, b) weight matrix W and bias b
             for each layer
    """
    inp_size = data.shape[2]
    out_size = labels.shape[2]
    if rng is None:
        rng = np.random.RandomState(1)
    print '.. model initialization'
    model = init_model(inp_size, [50] * kwargs['num_layers'], out_size,
                       rng=rng)

    print '.. starting training'
    for epoch in xrange(n_epochs):
        train_loss = 0.
        for batch, label in zip(data, labels):
            outputs, hiddens = forward_prop(batch, model)
            grads = backward_prop(batch, label, model, hiddens, outputs)
            train_loss += compute_loss(label, outputs)
            for (W, b), (grad_w, grad_b) in zip(model, grads):
                W -= learn_rate * grad_w
                b -= learn_rate * grad_b
        train_loss /= data.shape[0]
        valid_loss = 0.
        for batch, label in zip(valid_data, valid_labels):
            outputs, _ = forward_prop(batch, model)
            valid_loss += compute_loss(label, outputs)
        valid_loss /= valid_data.shape[0]
        print '.... epoch', epoch, 'validation loss', valid_loss, \
              'train loss', train_loss

    return model


def compute_misclass(label, outputs):
    return (np.argmax(label, axis=1) == np.argmax(outputs, axis=1)).sum()


def test(data, labels, model, **kwargs):
    """
    Computes testing error
    :param data: Testing data
    :param labels: Correct labels for `data`
    :param model: Trained model as described in function `train`
    :return: testing error
    """
    test_loss = 0.
    test_misclass = 0.
    for batch, label in zip(data, labels):
        outputs, _ = forward_prop(batch, model)
        test_loss += compute_loss(label, outputs)
        test_misclass += compute_misclass(label, outputs)
    test_loss /= data.shape[0]
    test_misclass /= data.shape[0] * data.shape[1]
    return test_loss, test_misclass


def init_model(num_feat, layer_sizes, num_out, rng=None, seed=1):
    model = []
    if rng is None:
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
        values += [hidden.copy()]
    V, c = model[-1]
    val = hidden.dot(V.T) + c.T
    outputs = softmax(val)
    values += [val.copy()]
    return outputs, values


def backprop_step(dEdY, W, gd, Y, X):
    deltas = (gd(Y) * dEdY)
    dEdX = deltas.dot(W)
    dEdW = deltas.T.dot(X) / deltas.shape[0]
    dEdb = deltas.mean(axis=0, keepdims=True).T
    return dEdX, dEdW, dEdb


def compute_loss(labels, outputs):
    return (-labels * np.log(outputs)).sum(axis=1).mean()


def backward_prop(data, labels, model, hiddens, outputs):
    """
    Performs backprop
    :param data: Batch of data (batch_index, data_index)
    :param labels: Labels for batch of data
    :param model: Model as described in `train`
    :param hiddens: Hidden variable values from forward propagation
    :param outputs: Outputs from forward propagation
    :return: All derivatives
    """
    # matrix (batch_index, output_index)
    err = ((-labels * (1. / outputs))[:, :, None] *
           softmax_grad(outputs)).sum(axis=1)
    err, w_diff, b_diff = backprop_step(err, model[-1][0],
                                        lambda x: np.ones_like(x),
                                        hiddens[-1], hiddens[-2])
    diffs = [(w_diff.copy(), b_diff.copy())]
    hiddens = [data] + hiddens
    for i, (W, b) in enumerate(reversed(model[:-1])):
        err, w_diff, b_diff = backprop_step(err, W, sigma_grad,
                                            hiddens[-i-2], hiddens[-i-3])
        diffs += [(w_diff.copy(), b_diff.copy())]

    return [x for x in reversed(diffs)]


def one_hot(data, out_size):
    out = np.zeros((len(data), out_size))
    out[(xrange(len(data)), data)] = 1.
    return out


def divide_by_batch(data, labels, batch_size):
    in_size = data.shape[1]
    out_size = labels.shape[1]
    return (data.reshape((-1, batch_size, in_size)),
            labels.reshape(-1, batch_size, out_size))


def load_data(filename, out_size, batch_size):
    datasets = []
    with gzip.open(filename, 'r') as fin:
        data = pkl.load(fin)
    for dataset in data:
        features, labels = dataset
        datasets += [divide_by_batch(features,
                                      one_hot(labels, out_size),
                                      batch_size)]
    return datasets


def parse_args():
    parser = argparse.ArgumentParser(description='Run MLP on MNIST')
    parser.add_argument('--filename',
                        default='/data/lisa/data/mnist/mnist.pkl.gz',
                        help='File which contains pickled dataset')
    parser.add_argument('--num_feat', type=int,
                        default=784,
                        help='Number of input features')
    parser.add_argument('--num_layers', type=int,
                        default=5,
                        help='Number of layers of MLP')
    parser.add_argument('--out_size', type=int,
                        default=10,
                        help='Size of output layer')
    parser.add_argument('--batch_size', type=int,
                        default=50,
                        help='Size of batch')
    parser.add_argument('--n_epochs', type=int,
                        default=50,
                        help='Number of epochs')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)
