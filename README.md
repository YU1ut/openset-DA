# openset-DA
This is an unofficial pytorch implementation of [Open Set Domain Adaptation by Backpropagation](https://arxiv.org/pdf/1804.10427.pdf). 

## Requirements
- Python 3.5+
- PyTorch 0.4
- torchvision
- scikit-learn

## Usage
Run SVHN -> MNIST
```
python train.py --task s2m --gpu <gpu_id>
```
Run USPS -> MNIST
```
python train.py --task u2m --gpu <gpu_id> --grl-rampup-epochs 50
```
Run MNIST -> USPS
```
python train.py --task m2u --gpu <gpu_id> --grl-rampup-epochs 30
```
For more details and parameters, please refer to --help option.

## Result

Please note that the parameters are not optimized.

SVHN -> MNIST

||OS|OS*|ALL|UNK| 
|:---|:---:|:---:|:---:|:---:|
|This Code|63.8|60.6|70.7|80.1|
|Paper|63.0|59.1|71.0|82.3|


USPS -> MNIST

||OS|OS*|ALL|UNK| 
|:---|:---:|:---:|:---:|:---:|
|This Code|84.8|85.8|82.8|79.5|
|Paper|92.3|91.2|94.4|97.6|


MNIST -> USPS

||OS|OS*|ALL|UNK| 
|:---|:---:|:---:|:---:|:---:|
|This Code|88.3|88.4|89.0|87.9|
|Paper|92.1|94.9|88.1|78.0|