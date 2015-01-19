__author__ = 'dima'

import numpy as np
import numpy.testing
import numpy.random

from mnist_mlp import init_model, forward_prop, backward_prop, compute_loss


def test_init_model():
    """Test model initialization"""
    model = init_model(5, [3, 2], 10)
    assert len(model) == 3
    assert model[0][0].shape == (3, 5)
    assert model[0][1].shape == (3, 1)
    assert model[1][0].shape == (2, 3)
    assert model[1][1].shape == (2, 1)
    assert model[2][0].shape == (10, 2)
    assert model[2][1].shape == (10, 1)


def test_forward_prop():
    """Test forward propagation"""
    model = init_model(5, [3, 2], 10)
    output, values = forward_prop(np.array([[0, 0, 0, 0, 0]]), model)
    assert output.shape == (1, 10)
    assert len(values) == 3


def test_forward_prop_batch():
    """Test forward propagation with bigger batch"""
    batch_size = 4
    inp_size = 5
    out_size = 10
    batch = np.zeros((batch_size, inp_size))
    model = init_model(inp_size, [3, 2], out_size)
    output, values = forward_prop(batch, model)
    assert output.shape == (batch_size, out_size)
    assert len(values) == 3


def test_backprop():
    """Test backprop"""
    batch_size = 1
    inp_size = 5
    out_size = 10
    batch = np.zeros((batch_size, inp_size))
    model = init_model(inp_size, [3, 2], out_size)
    output, hiddens = forward_prop(batch, model)

    label = np.zeros((batch_size, out_size))
    label[:, 0] = 1.

    diffs = backward_prop(batch, label, model, hiddens, output)


def test_backprop_batch():
    """Test backprop with batch"""
    batch_size = 4
    inp_size = 5
    out_size = 10
    batch = np.zeros((batch_size, inp_size))
    model = init_model(inp_size, [3, 2], out_size)
    output, hiddens = forward_prop(batch, model)

    label = np.zeros((batch_size, out_size))
    label[:, 0] = 1.

    ans = backward_prop(batch, label, model, hiddens, output)
    assert ans[0][0].shape == (batch_size,)


def test_gradients_v():
    """Test gradients for V"""
    delta = 1e-5
    batch_size = 4
    inp_size = 5
    out_size = 10
    batch = np.random.normal(0, 0.1, (batch_size, inp_size))
    model = init_model(inp_size, [3, 2], out_size)
    output, hiddens = forward_prop(batch, model)

    label = np.zeros((batch_size, out_size))
    label[:, 0] = 1.

    grads = backward_prop(batch, label, model, hiddens, output)

    shape = model[-1][0].shape
    num_grad = np.zeros(shape)
    m, n = shape
    for i in xrange(m):
        for j in xrange(n):
            outputs, _ = forward_prop(batch, model)
            loss = compute_loss(label, outputs)

            model[-1][0][i, j] += delta
            outputs_plus, _ = forward_prop(batch, model)
            loss_plus = compute_loss(label, outputs_plus)

            model[-1][0][i, j] -= delta
            num_grad[i, j] = (loss_plus - loss) / delta

    print grads[-1][0], num_grad
    np.testing.assert_allclose(grads[-1][0], num_grad, atol=1e-5)


def test_gradients_c():
    """Test gradients for c"""
    delta = 1e-5
    batch_size = 4
    inp_size = 5
    out_size = 10
    batch = np.random.normal(0, 0.1, (batch_size, inp_size))
    model = init_model(inp_size, [3, 2], out_size)
    output, hiddens = forward_prop(batch, model)

    label = np.zeros((batch_size, out_size))
    label[:, 0] = 1.

    grads = backward_prop(batch, label, model, hiddens, output)

    V = model[-1][0]
    diffs = np.zeros_like(V)
    for i in xrange(V.shape[0]):
        for j in xrange(V.shape[1]):
            outputs1, _ = forward_prop(batch, model)
            V[i, j] += delta
            model[-1][0] = V
            outputs2, _ = forward_prop(batch, model)
            V[i, j] -= delta
            model[-1][0] = V
            diffs[i, j] = (outputs2 - outputs1) / delta

    np.testing.assert_allclose(grads, diffs)


if __name__ == '__main__':
    test_backprop()
