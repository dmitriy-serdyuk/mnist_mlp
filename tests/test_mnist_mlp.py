__author__ = 'dima'

import numpy as np

from mnist_mlp import init_model, forward_prop, backward_prop


def test_init_model():
    """Test model initialization"""
    model = init_model(5, [3, 2], 10)
    assert len(model) == 3
    assert model[0][0].shape == (5, 3)
    assert model[0][1].shape == (3,)
    assert model[1][0].shape == (3, 2)
    assert model[1][1].shape == (2,)
    assert model[2][0].shape == (2, 10)
    assert model[2][1].shape == (10,)


def test_forward_prop():
    """Test forward propagation"""
    model = init_model(5, [3, 2], 10)
    output, hiddens = forward_prop(np.array([[0, 0, 0, 0, 0]]), model)
    assert output.shape == (10, 1)
    assert len(hiddens) == 2


def test_forward_prop_batch():
    """Test forward propagation with bigger batch"""
    batch_size = 4
    inp_size = 5
    out_size = 10
    batch = np.zeros((batch_size, inp_size))
    model = init_model(inp_size, [3, 2], out_size)
    output, hiddens = forward_prop(batch, model)
    assert output.shape == (out_size, batch_size)
    assert len(hiddens) == 2


def test_backprop():
    """Test backprop"""
    batch_size = 4
    inp_size = 5
    out_size = 10
    batch = np.zeros((batch_size, inp_size))
    model = init_model(inp_size, [3, 2], out_size)
    output, hiddens = forward_prop(batch, model)

    label = np.zeros((batch_size, out_size))
    label[:, 0] = 1.

    ans = backward_prop(batch, label, model, hiddens, output)
    assert ans.shape == (batch_size,)
