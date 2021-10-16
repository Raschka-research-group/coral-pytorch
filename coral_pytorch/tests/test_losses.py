# Sebastian Raschka 2020
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

from coral_pytorch.losses import coral_loss, corn_loss
import torch
import pytest


def test_coral_basic():

    levels = torch.tensor(
        [[1., 1., 0., 0.],
         [1., 0., 0., 0.],
         [1., 1., 1., 1.]])

    logits = torch.tensor(
        [[2.1, 1.8, -2.1, -1.8],
         [1.9, -1., -1.5, -1.3],
         [1.9, 1.8, 1.7, 1.6]])

    got_val = coral_loss(logits, levels, reduction=None)
    expect_val = torch.tensor([0.5370, 0.8951, 0.6441])
    assert torch.allclose(got_val, expect_val, rtol=1e-03, atol=1e-05)

    got_val = coral_loss(logits, levels, reduction='sum')
    expect_val = torch.tensor(2.0761)
    assert torch.allclose(got_val, expect_val, rtol=1e-03, atol=1e-05)

    got_val = coral_loss(logits, levels)
    expect_val = torch.tensor(0.6920)
    assert torch.allclose(got_val, expect_val, rtol=1e-03, atol=1e-05)


def test_coral_wrong_dim():
    levels = torch.tensor(
        [[1., 0., 0., 0.],
         [1., 1., 1., 1.]])

    logits = torch.tensor(
        [[2.1, 1.8, -2.1, -1.8],
         [1.9, -1., -1.5, -1.3],
         [1.9, 1.8, 1.7, 1.6]])

    with pytest.raises(ValueError) as excinfo:

        def f():
            coral_loss(logits, levels)

        f()
    assert "Please ensure that logits" in str(excinfo.value)


def test_coral_wrong_dim_2():
    levels = torch.tensor(
        [[1., 1., 0., 0.],
         [1., 0., 0., 0.],
         [1., 1., 1., 1.]])

    logits = torch.tensor(
        [[2.1, 1.8, -2.1, -1.8],
         [1.9, -1., -1.5, -1.3],
         [1.9, 1.8, 1.7, 1.6]])

    with pytest.raises(ValueError) as excinfo:

        def f():
            coral_loss(logits, levels, reduction='something')

        f()
    assert "Invalid value for `reduction`" in str(excinfo.value)


def test_corn_basic():

    logits = torch.tensor(
        [[2.1, 1.8, -2.1, -1.8],
         [1.9, -1., -1.5, -1.3],
         [1.9, 1.8, 1.7, 1.6]])

    y_train = torch.tensor([0, 1, 2])

    got_val = corn_loss(logits, y_train, num_classes=5)
    expect_val = torch.tensor([0.9657])
    assert torch.allclose(got_val, expect_val, rtol=1e-03, atol=1e-05)