# AutoProtoNet: Interpretability for Prototypical Networks
More information on this work is available on arXiv: https://arxiv.org/abs/2204.00929 

### Relevant files
- **models/autoprotonet.py** contains the `AutoProtoNetEmbedding` model
- **scripts/** contains code to train models and reproduce the models in the paper. For example, **scripts/train_mi_autoprotonet_5w5s.sh** contains the code to train AutoProtoNet 5-way 5-shot on miniImageNet.
- Our dataset for the experiment in [Section 4.2](https://arxiv.org/abs/2204.00929) (and Appendix A) is also available in this repo. The training split is in **train_novel_task_imgs_2** and the test split is in **test_novel_task_imgs_2**.
- **Paper - Prototypes Figure.ipynb** can be used to reproduce the figures of prototype images.
- **Paper - Case Study.ipynb** can be used to reproduce the case study of [Section 4.2](https://arxiv.org/abs/2204.00929).
- **pretrain_autoprotonet.py** can be used to pretrain on ImageNet.
- **train_autoprotonet.py** can be used to train AutoProtoNet from scratch on miniImageNet and other datasets.


### Prerequisites:
* Python2
* PyTorch
* CUDA


### Acknowledgements

This repo is a fork of "Adversarially Robust Few-Shot Learning: A Meta-Learning Approach" by Micah Goldblum, Liam Fowl, Tom Goldstein

This repository contains PyTorch code for adversarial querying with [ProtoNet](https://arxiv.org/abs/1703.05175), [R2-D2](https://arxiv.org/abs/1805.08136), and [MetaOptNet](https://arxiv.org/pdf/1904.03758.pdf).  Adversarial querying is an algorithm for producing robust meta-learners.  More can be found in our NeurIPS 2020 [paper](http://arxiv.org/abs/1910.00982).  We adapt models and data loading from [here](https://github.com/kjunelee/MetaOptNet).
