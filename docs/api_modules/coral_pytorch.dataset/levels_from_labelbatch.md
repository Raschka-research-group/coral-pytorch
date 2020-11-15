## levels_from_labelbatch

*levels_from_labelbatch(labels, num_classes, dtype=torch.float32)*

Converts a list of integer class label to extended binary label vectors

**Parameters**

- `labels` : list or 1D orch.tensor, shape=(num_labels,)

    A list or 1D torch.tensor with integer class labels
    to be converted into extended binary label vectors.


- `num_classes` : int

    The number of class clabels in the dataset. Assumes
    class labels start at 0. Determines the size of the
    output vector.


- `dtype` : torch data type (default=torch.float32)

    Data type of the torch output vector for the
    extended binary labels.

**Returns**

- `levels` : torch.tensor, shape=(num_labels, num_classes-1)


**Examples**

```
    >>> levels_from_labelbatch(labels=[2, 1, 4], num_classes=5)
    tensor([[1., 1., 0., 0.],
    [1., 0., 0., 0.],
    [1., 1., 1., 1.]])
```

