__author__ = 'serdyuk'

import numpy as np


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

if __name__ == '__main__':
    main()
