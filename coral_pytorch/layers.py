# Sebastian Raschka 2020
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import torch


class CoralLayer(torch.nn.Module):
    """ Implements CORAL layer described in

    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

    Parameters
    -----------
    size_in : int
        Number of input features for the inputs to the forward method, which
        are expected to have shape=(num_examples, num_features).

    num_classes : int
        Number of classes in the dataset.


    """
    def __init__(self, size_in, num_classes):
        super().__init__()
        self.size_in, self.size_out = size_in, 1

        self.coral_weights = torch.nn.Linear(self.size_in, 1, bias=False)
        self.coral_bias = torch.nn.Parameter(
             torch.zeros(num_classes-1).float())

    def forward(self, x):
        """
        Computes forward pass.

        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.

        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        return self.coral_weights(x) + self.coral_bias
