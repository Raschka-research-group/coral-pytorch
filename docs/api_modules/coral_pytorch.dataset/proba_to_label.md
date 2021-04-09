## proba_to_label

*proba_to_label(probas)*

Converts predicted probabilities from extended binary format
    to integer class labels

**Parameters**

- `probas` : torch.tensor, shape(n_examples, n_labels)

    Torch tensor consisting of probabilities returned by CORAL model.

**Examples**

```
    >>> # 3 training examples, 6 classes
    >>> probas = torch.tensor([[0.934, 0.861, 0.323, 0.492, 0.295],
    ...                        [0.496, 0.485, 0.267, 0.124, 0.058],
    ...                        [0.985, 0.967, 0.920, 0.819, 0.506]])
    >>> proba_to_label(probas)
    tensor([2, 0, 5])
```

