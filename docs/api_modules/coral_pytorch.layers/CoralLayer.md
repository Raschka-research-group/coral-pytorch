## CoralLayer

*CoralLayer(size_in, num_classes)*

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

