<img src="docs/img/corn-coral-logo-alpha.png" width=300>

**CORAL & CORN implementations for ordinal regression with deep neural networks.**


[![PyPI version](https://badge.fury.io/py/coral-pytorch.svg)](https://badge.fury.io/py/coral-pytorch)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Raschka-research-group/coral-pytorch/blob/main/LICENSE.txt)
![Python 3](https://img.shields.io/badge/python-3-blue.svg)

<br>

**Documentation: [https://Raschka-research-group.github.io/coral-pytorch](https://Raschka-research-group.github.io/coral-pytorch)**

---



## Installation



```bash
pip install coral-pytorch
```





## About  

CORAL (COnsistent RAnk Logits) and CORN (Conditional Ordinal Regression for Neural networks) are methods for ordinal regression with deep neural networks, which address the rank inconsistency issue of other ordinal regression frameworks.

<img src="docs/img/figure1.jpg" width=400>

Originally, developed this method in the context of age prediction from face images. Our approach was evaluated on several face image datasets for age prediction using ResNet-34, but it is compatible with other state-of-the-art deep neural networks.

This repository implements the CORAL and CORN functionality (neural network layer, loss function, and dataset utilities) for convenient use. Examples are provided via the "Tutorials" that can be found on the documentation website at [https://Raschka-research-group.github.io/coral-pytorch](https://Raschka-research-group.github.io/coral-pytorch).

If you are looking for the orginal implementation, training datasets, and training log files corresponding to the paper, you can find these here: 

- CORAL: [https://github.com/Raschka-research-group/coral-cnn](https://github.com/Raschka-research-group/coral-cnn).
- CORN: [https://github.com/Raschka-research-group/corn-ordinal-neuralnet](https://github.com/Raschka-research-group/corn-ordinal-neuralnet) 



---

## Cite as

If you use CORAL or CORN  as part of your workflow in a scientific publication, please consider citing the corresponding paper:

**CORAL**

- Wenzhi Cao, Vahid Mirjalili, and Sebastian Raschka (2020).  *Rank Consistent Ordinal Regression for Neural Networks with Application to Age Estimation*. Pattern Recognition Letters 140, pp. 325-331; [https://doi.org/10.1016/j.patrec.2020.11.008](https://doi.org/10.1016/j.patrec.2020.11.008).



```
@article{coral2020,
title={Rank consistent ordinal regression for neural networks with application to age estimation},
journal={Pattern Recognition Letters},
volume={140},
pages={325-331},
year={2020},
issn={0167-8655},
doi={https://doi.org/10.1016/j.patrec.2020.11.008},
url={http://www.sciencedirect.com/science/article/pii/S016786552030413X},
author={Wenzhi Cao and Vahid Mirjalili and Sebastian Raschka}
}
```

**CORN**

- Xintong Shi, Wenzhi Cao, and Sebastian Raschka (2021). Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities. Arxiv preprint;  [https://arxiv.org/abs/2111.08851](https://arxiv.org/abs/2111.08851)

```
@misc{shi2021deep,
      title={Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities}, 
      author={Xintong Shi and Wenzhi Cao and Sebastian Raschka},
      year={2021},
      eprint={2111.08851},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
