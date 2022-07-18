# Release Notes

The changelog for the current development version is available at
[https://github.com/raschka-research-group/coral_pytorch/blob/main/docs/CHANGELOG.md](https://github.com/raschka-research-group/coral_pytorch/blob/main/docs/CHANGELOG.md.



### 1.4.0 (07-17-2022)

##### Downloads

- [Source code (zip)](https://github.com/raschka-research-group/coral_pytorch/archive/v1.4.0.zip)
- [Source code (tar.gz)](https://github.com/raschka-research-group/coral_pytorch/archive/v1.4.0.tar.gz)

##### New Features

- Adds object-oriented versions of the losses: `coral_pytorch.losses.CoralLoss` and `coral_pytorch.losses.CornLoss`.

##### Changes

- - 

##### Bug Fixes

- -


### 1.3.0 (07-16-2022)

##### Downloads

- [Source code (zip)](https://github.com/raschka-research-group/coral_pytorch/archive/v1.3.0.zip)
- [Source code (tar.gz)](https://github.com/raschka-research-group/coral_pytorch/archive/v1.3.0.tar.gz)

##### New Features

- -

##### Changes

- -

##### Bug Fixes

- Fixes a bug where the normalization of the `corn_loss` different from the one proposed in the original paper. ([#22](https://github.com/Raschka-research-group/coral-pytorch/pull/22/files)) 


### 1.2.0 (11-17-2021)

##### Downloads

- [Source code (zip)](https://github.com/raschka-research-group/coral_pytorch/archive/v1.2.0.zip)
- [Source code (tar.gz)](https://github.com/raschka-research-group/coral_pytorch/archive/v1.2.0.tar.gz)

##### New Features

- Add CORN loss corresponding to the manuscript, "[Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities](https://arxiv.org/abs/2111.08851)"

##### Changes

- - 

##### Bug Fixes

- -



### 1.1.0 (04/08/2021)

##### Downloads

- [Source code (zip)](https://github.com/raschka-research-group/coral_pytorch/archive/v1.1.0.zip)
- [Source code (tar.gz)](https://github.com/raschka-research-group/coral_pytorch/archive/v1.1.0.tar.gz)

##### New Features

- -

##### Changes

- By default, bias units are now preinitialized to descending values in [0, 1] range (instead of all zero values), which results in faster training and better generalization performance. (PR [#5](https://github.com/Raschka-research-group/coral_pytorch/pull/5))

##### Bug Fixes

- -


### 1.0.0 (11/15/2020)

##### Downloads

- [Source code (zip)](https://github.com/raschka-research-group/coral_pytorch/archive/v1.0.0.zip)
- [Source code (tar.gz)](https://github.com/raschka-research-group/coral_pytorch/archive/v1.0.0.tar.gz)

##### New Features

- First release.

##### Changes

- First release.

##### Bug Fixes

- First release.