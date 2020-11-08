## label_to_levels

*label_to_levels(label, num_classes, dtype=torch.float32)*

Converts integer class label to extended binary label vector

**Parameters**

- `label` : int

    Class label to be converted into a extended
    binary vector. Should be smaller than num_classes-1.


- `num_classes` : int

    The number of class clabels in the dataset. Assumes
    class labels start at 0. Determines the size of the
    output vector.


- `dtype` : torch data type (default=torch.float32)

    Data type of the torch output vector for the
    extended binary labels.

**Returns**

- `levels` : torch.tensor, shape=(num_classes-1,)

    Extended binary label vector. Type is determined
    by the `dtype` parameter.

**Examples**

```
    >>> label_to_levels(0, num_classes=5)
    tensor([0., 0., 0., 0.])
    >>> label_to_levels(1, num_classes=5)
    tensor([1., 0., 0., 0.])
    >>> label_to_levels(3, num_classes=5)
    tensor([1., 1., 1., 0.])
    >>> label_to_levels(4, num_classes=5)
    tensor([1., 1., 1., 1.])
```

