# Sebastian Raschka 2020
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

from coral_pytorch.dataset import label_to_levels
import torch
import pytest


def test_wrong_numclasses():

    with pytest.raises(ValueError) as excinfo:

        def f():
            label_to_levels(5, num_classes=5)

        f()
    assert "Class label must be smaller" in str(excinfo.value)


def test_basic():
    got = label_to_levels(0, num_classes=5)
    expect = torch.tensor([0., 0., 0., 0.])
    assert torch.all(torch.eq(got, expect)).item()

    got = label_to_levels(1, num_classes=5)
    expect = torch.tensor([1., 0., 0., 0.])
    assert torch.all(torch.eq(got, expect)).item()

    got = label_to_levels(2, num_classes=5)
    expect = torch.tensor([1., 1., 0., 0.])
    assert torch.all(torch.eq(got, expect)).item()

    got = label_to_levels(3, num_classes=5)
    expect = torch.tensor([1., 1., 1., 0.])
    assert torch.all(torch.eq(got, expect)).item()

    got = label_to_levels(4, num_classes=5)
    expect = torch.tensor([1., 1., 1., 1.])
    assert torch.all(torch.eq(got, expect)).item()


def test_torchinput():
    got = label_to_levels(torch.tensor(3), num_classes=5)
    expect = torch.tensor([1., 1., 1., 0.])
    assert torch.all(torch.eq(got, expect)).item()
