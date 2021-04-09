coral_pytorch version: 1.1.0
## CoralLayer

*CoralLayer(size_in, num_classes, preinit_bias=True)*

Implements CORAL layer described in

    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
    with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008

**Parameters**

- `size_in` : int

    Number of input features for the inputs to the forward method, which
    are expected to have shape=(num_examples, num_features).


- `num_classes` : int

    Number of classes in the dataset.


- `preinit_bias` : bool (default=True)

    If true, it will pre-initialize the biases to descending values in
    [0, 1] range instead of initializing it to all zeros. This pre-
    initialization scheme results in faster learning and better
    generalization performance in practice.

