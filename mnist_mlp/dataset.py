__author__ = 'dima'

import gzip
import cPickle as pkl

import numpy as np


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
