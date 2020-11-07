# Sebastian Raschka 2020
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import torch


class CoralLayer(torch.nn.Module):
    """ CORAL layer """
    def __init__(self, size_in, num_classes):
        super().__init__()
        self.size_in, self.size_out = size_in, 1

        self.coral_weights = torch.nn.Linear(self.size_in, 1, bias=False)
        self.coral_bias = torch.nn.Parameter(
            torch.zeros(num_classes-1).float())

    def forward(self, x):
        return self.coral_weights(x) + self.coral_bias