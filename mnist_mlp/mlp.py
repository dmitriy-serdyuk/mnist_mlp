__author__ = 'dima'

import numpy as np

from utils import softmax, softmax_grad, sigma, sigma_grad, pairwise


def init_model(num_feat, layer_sizes, num_out, rng=None, seed=1):
    """
    Randomly initializes a model

    :param num_feat: Number of features
    :param layer_sizes: An array of layer sizes
    :param num_out: Number of outputs
    :param rng: Random number generator
    :param seed: Seed to create a random number generator if `rng` was not
           passed
    :return: Model
    """
    model = []
    if rng is None:
        rng = np.random.RandomState(seed)
    layer_sizes = [num_feat] + layer_sizes + [num_out]
    for size in pairwise(layer_sizes):
        W = rng.normal(scale=0.1, size=size).T
        b = rng.normal(scale=0.1, size=(size[1], 1))
        model += [(W, b)]
    return model


def forward_prop(feat_batch, model):
    """
    Performs forward propagation and returns the last layer values
    :param feat_batch: Batch of data
    :param model: Model to use as described in `train`
    :return: List of outputs for each data point from `data` and values
             obtained
    """
    hidden = feat_batch
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
    """
    Performs one step of backpropagation

    :param dEdY: Previous gradient
    :param W: Weight matrix for this iteration
    :param gd: Gradient function with respect to `Y`
    :param Y: Output of the layer
    :param X: Input of the layer
    :return: Gradient, gradient of `W`, gradient of bias `b`
    """
    deltas = gd(Y) * dEdY
    dEdX = deltas.dot(W)
    dEdW = deltas.T.dot(X) / deltas.shape[0]
    dEdb = deltas.mean(axis=0, keepdims=True).T
    return dEdX, dEdW, dEdb


def backward_prop(feats, labels, model, hiddens, outputs):
    """
    Performs backprop
    :param feats: Batch of features (batch_index, data_index)
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
    hiddens = [feats] + hiddens
    for i, (W, b) in enumerate(reversed(model[:-1])):
        err, w_diff, b_diff = backprop_step(err, W, sigma_grad,
                                            hiddens[-i-2], hiddens[-i-3])
        diffs += [(w_diff.copy(), b_diff.copy())]

    return [x for x in reversed(diffs)]
