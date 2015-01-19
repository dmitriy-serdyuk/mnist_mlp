__author__ = 'dima'
import time
from itertools import tee, izip

import numpy as np


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print '.. %s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
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


def compute_misclass(label, outputs):
    return (np.argmax(label, axis=1) != np.argmax(outputs, axis=1)).sum()


def compute_loss(labels, outputs):
    return (-labels * np.log(outputs)).sum(axis=1).mean()


