# FNBench
The initial version of official codes for our paper [FNBench: Benchmarking Robust Federated Learning against Noisy Labels](https://arxiv.org/abs/2505.06684). It serves as a benchmark platform for researchers to evaluate robust federated learning algorithms against noisy labels. If you have any questions, please feel free to contact me: ) 

## Previous Abstract
<!-- TBD. -->

![framework](TDSC.jpg)

## Datasets
For vision tasks, we use CIFAR-10, CIFAR-100, CIFAR-10-N, CIFAR-100-N, and one large-scale online clothing datasets Clothing1M for evaluation. 
For language tasks, we use AGNews for evaluation.
We list a table as below:

| Dataset | Model | #Classes | #Train Set | #Test Set | Label Noise Pattern | Extra Information  |
| :-----: | :------: | :------: | :--------: | :-------: | :----------------: | :----------------: |
| CIFAR-10 | ResNet-18 |   10    |   50,000   |   10,000  | Synthetic | -  |
| CIFAR-100| ResNet-32 |   100   |   50,000   |   10,000  | Synthetic | -  |
| AGNews |   FastText | 4     |  120,000  |   7,600  | Synthetic | [link](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)  |
| CIFAR-10-N| ResNet-18 |  10 |   50,000   |   10,000  |  Human Annotation Errors | [link](https://github.com/UCSC-REAL/cifar-10-100n)  |
| CIFAR-100-N | ResNet-32 | 100  |   50,000   |   10,000  | Human Annotation Errors  | [link](https://github.com/UCSC-REAL/cifar-10-100n)  |
| Clothing1M| ResNet-50 |  14 | 1,000,000  |   10,000 | Systematic Errors | [link](https://github.com/Cysu/noisy_label)  |


For the last three datasets, considering related copyrights, please refer to the corresponding links for dataset requisition. Meanwhile, we provide an implementation code to experiment on `Clothing1M` in this [codebase](https://github.com/Sprinter1999/Clothing1M_FedAvg).

## Baselines

- **General FL methods**: FedAvg, FedProx, FedExP
- **Robust FL methods**: TrimmedMean, Krum, Median, RFA
- **General Noisy Label Learning (NLL) methods**: Co-teaching, Co-teaching+, SymmetricCE, SELFIE, Joint Optim, DivideMix
- **Federated Nosy Label Learning (FNLL) methods**: RobustFL, FedLSR, FedNoRo, [FedRN](https://github.com/ElvinKim/FedRN), FedELC，FedELR

For FedELR re-implementation, please kindly refer to [the original paper](https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=4995227) since it has no official implementation. We also provide one implementation of one new work [MaskedOptim](https://github.com/Sprinter1999/MaskedOptim).


## Example for Usage
> bash eval_fedavg.sh

please refer to the `./utils/options.py` for more details. For CIFAR-N experiments, please use another two main files (like main_fed_LNL_cifar10N.py). We recommend you to experiment on Nvidia 3090 (24G) GPUS or more advanced GPUs. 





## Awesome Resources
We recommend some useful related resources to further provide several relevant directions for future study.

| Name | Summary | Code Link |
| :---: | :---: | :---: |
| FedNoisy | Recommended codebase for FNLL research | [Link](https://github.com/SMILELab-FL/FedNoisy) |
| Clothing1M_FedAvg | Recommended codebase for FedAvg on Clothing1M | [Link](https://github.com/Sprinter1999/Clothing1M_FedAvg) |
| FedRN | Referred codebase for implementation | [Link](https://github.com/ElvinKim/FedRN) |
| HAR Datasets (ACM Mobisys) | Recommended time-series Datasets | [Link](https://github.com/xmouyang/FL-Datasets-for-HAR) |
| FedDSHAR (FGCS) | Recommended work to tackle noisy labels for time-series | [Link](https://github.com/coke2020ice/FedDSHAR) |
| FedNed (AAAI) | Recommended work to tackle extreme noisy clients | [Link](https://github.com/linChen99/FedNed) |
| FedAAAI (AAAI) | Recommended work to tackle label noise for image segmentation  | [Link](https://github.com/wnn2000/FedAAAI) |
| Buffalo (ACM CIKM) | Recommended work to tackle modality hetogeneity | [Link](https://github.com/beiyuouo/Buffalo) |
| Twin-sight (ICLR) | Recommended work to tackle semi-supervised learning | [Link](https://github.com/visitworld123/Twin-sight) |

## Acknowledgements
In recent years, we have proposed `FedLSR (ACM CIKM'22)`, `FedNoRo (IJCAI'23)`, `FedELC (ACM CIKM'24)`, `FedDSHAR (FGCS)` ,`Dual Optim (under review)` and this benchmark study `FNBench (IEEE TDSC, under review)`. 
We benifit from many well-organized open-source projects.
We encourage and hope more efforts can be made to study the noisy label issue in diverse research domains. If you find our work helpful, please consider following citations.

By the way, collaborations and pull requests are always welcome! If you have any questions or suggestions, please feel free to contact me : )

```bibtex
@article{Jiang_2024,
title={FNBench: Benchmarking Robust Federated Learning against Noisy Labels},
url={http://dx.doi.org/10.36227/techrxiv.172503083.36644691/v1},
DOI={10.36227/techrxiv.172503083.36644691/v1},
publisher={Institute of Electrical and Electronics Engineers (IEEE)},
author={Jiang, Xuefeng and Li, Jia and Wu, Nannan and Wu, Zhiyuan and Li, Xujing and Sun, Sheng and Xu, Gang and Wang, Yuwei and Li, Qi and Liu, Min},
year={2024},
}

@article{jiang2024tackling,
  title={Tackling Noisy Clients in Federated Learning with End-to-end Label Correction},
  author={Jiang, Xuefeng and Sun, Sheng and Li, Jia and Xue, Jingjing and Li, Runhan and Wu, Zhiyuan and Xu, Gang and Wang, Yuwei and Liu, Min},
  journal={arXiv preprint arXiv:2408.04301},
  year={2024}
}

@inproceedings{wu2023fednoro,
  title={FedNoRo: towards noise-robust federated learning by addressing class imbalance and label noise heterogeneity},
  author={Wu, Nannan and Yu, Li and Jiang, Xuefeng and Cheng, Kwang-Ting and Yan, Zengqiang},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence},
  pages={4424--4432},
  year={2023}
}

@inproceedings{kim2022fedrn,
  title={FedRN: Exploiting k-reliable neighbors towards robust federated learning},
  author={Kim, SangMook and Shin, Wonyoung and Jang, Soohyuk and Song, Hwanjun and Yun, Se-Young},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={972--981},
  year={2022}
}

@inproceedings{jiang2022towards,
  title={Towards federated learning against noisy labels via local self-regularization},
  author={Jiang, Xuefeng and Sun, Sheng and Wang, Yuwei and Liu, Min},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={862--873},
  year={2022}
}
```
