__author__ = 'serdyuk'

import numpy as np
from itertools import tee, izip


def sigma(x):
    return 1. / (1. + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)


def main():
    data = None
    labels = None
    model = train(data, labels)
    test(data, labels, model)
    pass


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
        W = rng.normal(scale=0.1, size=size)
        b = rng.normal(scale=0.1, size=size[1])
        model += [(W, b)]
    return model


def forward_prop(batch, model):
    """
    Performs forward propagation and returns the last layer values
    :param batch: Batch of data
    :param model: Model to use as described in `train`
    :return: List of outputs for each data point from `data`
    """
    outputs = []
    hidden = batch.T
    for W, b in model[:-1]:
        hidden = sigma(W.T.dot(hidden) + b[:, None])
    V, c = model[-1]
    outputs += softmax(V.T.dot(hidden) + c[:, None])
    return outputs


def backward_prop(data, labels, model):
    """
    Performs backprop
    :param data: Batch of data
    :param labels: Labels for batch of data
    :param model: Model as described in `train`
    :return:
    """
    pass

if __name__ == '__main__':
    main()
