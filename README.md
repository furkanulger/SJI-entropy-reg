# reg_skewJSD
Entropy regularization with skewed Jensen-Shannon Divergence


This repository contains PyTorch implementation for the paper titled "Fine-grained Classification of Solder Joints with α-skew Jensen-Shannon Divergence.".
If you find this repository useful, please give reference to the paper:

Ulger, Furkan, et al. "Fine-grained Classification of Solder Joints with α-skew Jensen-Shannon Divergence." arXiv preprint arXiv:2209.09857 (2022).
[[arXiv]](https://arxiv.org/abs/2209.09857)

## Abstract
Solder joint inspection (SJI) is a critical process in the production of printed circuit boards (PCB). Detection of solder errors during SJI is quite challenging as the solder joints have very small sizes and can take various shapes. In this study, we first show that solders have low feature diversity, and that the SJI can be carried out as a fine-grained image classification task which focuses on hard-to-distinguish object classes. To improve the fine-grained classification accuracy, penalizing confident model predictions by maximizing entropy was found useful in the literature. Inline with this information, we propose using the {\alpha}-skew Jensen-Shannon divergence ({\alpha}-JS) for penalizing the confidence in model predictions. We compare the {\alpha}-JS regularization with both existing entropyregularization based methods and the methods based on attention mechanism, segmentation techniques, transformer models, and specific loss functions for fine-grained image classification tasks. We show that the proposed approach achieves the highest F1-score and competitive accuracy for different models in the finegrained solder joint classification task. Finally, we visualize the activation maps and show that with entropy-regularization, more precise class-discriminative regions are localized, which are also more resilient to noise. Code will be made available here upon acceptance.

<img width="434" alt="CAM" src="https://github.com/furkanulger/reg_skewJSD/blob/main/CAM.PNG">
