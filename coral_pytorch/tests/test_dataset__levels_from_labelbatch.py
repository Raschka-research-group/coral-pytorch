# Sebastian Raschka 2020
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

from coral_pytorch.dataset import levels_from_labelbatch
import torch


def test_basic():
    num_classes = 5
    class_labels = [2, 1, 4]

    expect = torch.tensor(
        [[1., 1., 0., 0.],
         [1., 0., 0., 0.],
         [1., 1., 1., 1.]])

    got = levels_from_labelbatch(class_labels, num_classes)
    assert torch.all(torch.eq(got, expect)).item()


def test_torchinput():
    num_classes = 5
    class_labels = torch.tensor([2, 1, 4])

    expect = torch.tensor(
        [[1., 1., 0., 0.],
         [1., 0., 0., 0.],
         [1., 1., 1., 1.]])

    got = levels_from_labelbatch(class_labels, num_classes)
    assert torch.all(torch.eq(got, expect)).item()
