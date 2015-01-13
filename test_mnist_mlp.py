__author__ = 'dima'

import numpy as np

from mnist_mlp import init_model, forward_prop


def test_forward_prob():
    """Check if no errors"""
    model = init_model(5, [3, 2], 10)
    forward_prop(np.array([[0, 0, 0, 0, 0]]), model)


def test_init_model():
    model = init_model(5, [3, 2], 10)
    assert len(model) == 3
    assert model[0][0].shape == (5, 3)
    assert model[0][1].shape == (3,)
    assert model[1][0].shape == (3, 2)
    assert model[1][1].shape == (2,)
    assert model[2][0].shape == (2, 10)
    assert model[2][1].shape == (10,)
