# Sebastian Raschka 2020
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import torch.nn.functional as F
import torch


def coral_loss(logits, levels, imp):
    val = (-torch.sum((F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels))*imp,
           dim=1))
    return torch.mean(val)