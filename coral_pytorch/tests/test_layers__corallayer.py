# Sebastian Raschka 2020
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import torch
from coral_pytorch.layers import CoralLayer


def test_basic():
    class MLP(torch.nn.Module):

        def __init__(self, num_classes):
            super(MLP, self).__init__()

            self.features = torch.nn.Sequential(
                torch.nn.Linear(75, 25),
                torch.nn.Linear(25, 10))

            self.fc = CoralLayer(size_in=10, num_classes=num_classes)

        def forward(self, x):
            x = self.features(x)

            #######################
            ##### CORAL LAYER #####
            ###--------------------------------------------------------------------###
            logits =  self.fc(x)
            probas = torch.sigmoid(logits)
            ###--------------------------------------------------------------------###

            return logits, probas


    model = MLP(num_classes=10)

    logits, probas = model(torch.rand(100, 75))

    assert logits.shape == torch.Size([100, 9])
    assert probas.shape == torch.Size([100, 9])
