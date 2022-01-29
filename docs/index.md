
<img src="./img/corn-coral-logo-alpha.png" width=300>

**CORAL implementation for ordinal regression with deep neural networks.**


[![PyPI version](https://badge.fury.io/py/coral-pytorch.svg)](https://badge.fury.io/py/coral-pytorch)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rasbt/coral_pytorch/blob/master/LICENSE)
![Python 3](https://img.shields.io/badge/python-3-blue.svg)

<br>

---

## About  

CORAL (COnsistent RAnk Logits) and CORN (Conditional Ordinal Regression for Neural networks) are methods for ordinal regression with deep neural networks, which address the rank inconsistency issue of other ordinal regression frameworks.

<img src="img/figure1.jpg" width=400>

Originally, developed this method in the context of age prediction from face images. Our approach was evaluated on several face image datasets for age prediction using ResNet-34, but it is compatible with other state-of-the-art deep neural networks.

This repository implements the CORAL and CORN functionality (neural network layer, loss function, and dataset utilities) for convenient use. Examples are provided via the "Tutorials" that can be found on the documentation website at [https://Raschka-research-group.github.io/coral_pytorch](https://Raschka-research-group.github.io/coral_pytorch).

If you are looking for the orginal implementation, training datasets, and training log files corresponding to the paper, you can find these here: 

- CORAL: [https://github.com/Raschka-research-group/coral-cnn](https://github.com/Raschka-research-group/coral-cnn).
- CORN: [https://github.com/Raschka-research-group/corn-ordinal-neuralnet](https://github.com/Raschka-research-group/corn-ordinal-neuralnet) 



### References



**CORAL**

- Wenzhi Cao, Vahid Mirjalili, and Sebastian Raschka (2020).  *Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation*. Pattern Recognition Letters 140, pp. 325-331; [https://doi.org/10.1016/j.patrec.2020.11.008](https://doi.org/10.1016/j.patrec.2020.11.008).


**CORN**

- Xintong Shi, Wenzhi Cao, and Sebastian Raschka (2021). Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities. Arxiv preprint;  [https://arxiv.org/abs/2111.08851](https://arxiv.org/abs/2111.08851)