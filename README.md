# Fine-grained Classification of Solder Joints with α-skew Jensen-Shannon Divergence


This repository contains PyTorch implementation for the paper titled "Fine-grained Classification of Solder Joints with α-skew Jensen-Shannon Divergence.".
If you find this repository useful, please give reference to the paper:

F. Ulger, S. E. Yuksel, A. Yilmaz and D. Gokcen, "Fine-Grained Classification of Solder Joints With α-Skew Jensen–Shannon Divergence," in IEEE Transactions on Components, Packaging and Manufacturing Technology, vol. 13, no. 2, pp. 257-264, Feb. 2023, doi: 10.1109/TCPMT.2023.3249193.
[[IEEE]](https://ieeexplore.ieee.org/abstract/document/10054132)

## Abstract
Solder joint inspection (SJI) is a critical process in the production of printed circuit boards (PCB). Detection of solder errors during SJI is quite challenging as the solder joints have very small sizes and can take various shapes. In this study, we first show that solders have low feature diversity, and that the SJI can be carried out as a fine-grained image classification task which focuses on hard-to-distinguish object classes. To improve the fine-grained classification accuracy, penalizing confident model predictions by maximizing entropy was found useful in the literature. Inline with this information, we propose using the {\alpha}-skew Jensen-Shannon divergence ({\alpha}-JS) for penalizing the confidence in model predictions. We compare the {\alpha}-JS regularization with both existing entropyregularization based methods and the methods based on attention mechanism, segmentation techniques, transformer models, and specific loss functions for fine-grained image classification tasks. We show that the proposed approach achieves the highest F1-score and competitive accuracy for different models in the finegrained solder joint classification task. Finally, we visualize the activation maps and show that with entropy-regularization, more precise class-discriminative regions are localized, which are also more resilient to noise.

<img width="434" alt="CAM" src="https://github.com/furkanulger/reg_skewJSD/blob/main/CAM.PNG">


## Installation
```
git clone https://github.com/furkanulger/SJI-entropy-reg.git
cd SJI-entropy-reg/
pip install -r requirements.txt
```
## About the Dataset
The dataset used in the paper is private, therefore you need to use your own dataset. The dataset folder structure should be as follows:
```
    .
    ├── dataset                   
    │   ├── train 
    │   ├── validation        
    │   └── test    
    │        ├── normal
    │        ├── defective
    │            ├──image1.png
    │            ├──image2.png
    │                .
    │                .    
    └── ...
    
    
```   
## Usage

Type -h to get description of the parameters for the script as follows:

<img width="693" alt="help" src="https://user-images.githubusercontent.com/50952046/211635714-3a9f720f-d63d-4125-ab6d-d70b47fcaf74.PNG">

To train GoogleNet with alpha-skew Jensen-Shannon divergence where a= 0.1, batch size = 64, learning rate= 1e-4, number of epochs= 100:
```python
cd SJI_entropy_reg/
python skewed_JSdivergence.py -t train -m <model> -c <number_of_classes>(default:2) -a <skewness_value>(default:0.5) -lr <learning rate>(default:1e-4) -e <num_of_epochs>(default:150)
```    
Example:
```python
python skewed_JSdivergence.py -t train -m googlenet -a 0.1 -bs 64 -lr 1e-4 -e 100
```  
To test the trained model with alpha-skew Jensen-Shannon divergence where a= 0.1:
```python
python skewed_JSdivergence.py -t test -m googlenet -a 0.1 -p results/skewed_JSD_model_.pth
```  
