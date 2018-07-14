# openset-DA
This is an unofficial pytorch implementation of [Open Set Domain Adaptation by Backpropagation](https://arxiv.org/pdf/1804.10427.pdf). 

Just on SVHN -> MNIST so far.

Please note that this is an ongoing project.

## Requirements
- Python 3.5+
- PyTorch 0.4
- torchvision
- scikit-learn

## Usage
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py
```

## Result
This Code: OS 67.8 OS* 65.1 ALL 73.7 UNK 81.3

Paper: OS 63.0 OS* 59.1 ALL 71.0 UNK 82.3