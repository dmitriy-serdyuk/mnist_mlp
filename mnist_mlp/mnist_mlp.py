__author__ = 'serdyuk'

import cPickle as pkl
import argparse

import numpy as np

from utils import timing, compute_loss, compute_misclass
from dataset import load_data
from mlp import forward_prop, backward_prop, init_model


def main(filename, out_size, batch_size, **kwargs):
    train_data, valid_data, test_data = load_data(filename, out_size,
                                                  batch_size)

    train_data, train_labels = train_data
    valid_data, valid_labels = valid_data
    test_data, test_labels = test_data

    rng = np.random.RandomState(1)
    model = train(train_data, train_labels, valid_data, valid_labels, rng=rng,
                  **kwargs)
    with open('model.pkl', 'w') as fout:
        pkl.dump(model, fout)

    loss, misclass = test(test_data, test_labels, model)
    print '.. testing error', loss, 'misclassification', misclass


@timing
def train(data, labels, valid_data, valid_labels, n_epochs,
          learn_rate, num_layers, n_hiddens, rng, **kwargs):
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
    if 'model_file' in kwargs:
        print '.. loading model'
        with open(kwargs['model_file'], 'r') as fin:
            model = pkl.load(fin)
    else:
        print '.. model initialization'
        model = init_model(inp_size, [n_hiddens] * num_layers, out_size,
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
        with open('model.pkl', 'w') as fout:
            pkl.dump(model, fout)
        print '.... model saved'

    return model


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


def parse_args():
    parser = argparse.ArgumentParser(description='Run MLP on MNIST')
    parser.add_argument('--filename',
                        default='/data/lisa/data/mnist/mnist.pkl.gz',
                        help='File which contains pickled dataset')
    parser.add_argument('--num_feat', type=int,
                        default=784,
                        help='Number of input features')
    parser.add_argument('--num_layers', type=int,
                        default=2,
                        help='Number of layers of MLP')
    parser.add_argument('--out_size', type=int,
                        default=10,
                        help='Size of output layer')
    parser.add_argument('--batch_size', type=int,
                        default=50,
                        help='Size of batch')
    parser.add_argument('--n_epochs', type=int,
                        default=100,
                        help='Number of epochs')
    parser.add_argument('--learn_rate', type=float,
                        default=1e-1,
                        help='Learning rate')
    parser.add_argument('--n_hiddens', type=int,
                        default=500,
                        help='Number of hidden units')
    parser.add_argument('--model_file',
                        help='Model file')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)
