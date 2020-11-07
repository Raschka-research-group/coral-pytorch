# Sebastian Raschka 2020
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import torch


def label_to_levels(label, num_classes, dtype=torch.float32):
    if isinstance(label, torch.Tensor):
        int_label = label.item()
    else:
        int_label = label

    levels = [1]*int_label + [0]*(num_classes - 1 - int_label)
    levels = torch.tensor(levels, dtype=dtype)
    return levels


def levels_from_labelbatch(labels, num_classes):
    levels = []
    for label in labels:
        levels_from_label = label_to_levels(
            label=label, num_classes=num_classes)
        levels.append(levels_from_label)

    levels = torch.stack(levels)
    return levels
