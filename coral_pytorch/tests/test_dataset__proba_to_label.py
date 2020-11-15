# Sebastian Raschka 2020
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

from coral_pytorch.dataset import proba_to_label
import torch


def test_basic():
    # 3 training examples, 6 classes
    probas = torch.tensor([[0.934, 0.861, 0.323, 0.492, 0.295],
                           [0.496, 0.485, 0.267, 0.124, 0.058],
                           [0.985, 0.967, 0.920, 0.819, 0.506]])

    got = proba_to_label(probas)
    expect = torch.tensor([2, 0, 5])
    assert torch.all(torch.eq(got, expect)).item()
