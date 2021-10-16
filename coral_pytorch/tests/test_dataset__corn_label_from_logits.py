# Sebastian Raschka 2020
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

from coral_pytorch.dataset import corn_label_from_logits
import torch


def test_basic():
    # 2 training examples, 5 classes
    logits = torch.tensor([[14.152, -6.1942, 0.47710, 0.96850],
                           [65.667, 0.303, 11.500, -4.524]])
    got = corn_label_from_logits(logits)

    expect = torch.tensor([1, 3])
    assert torch.all(torch.eq(got, expect)).item()
