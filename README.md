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
||OS|OS*|ALL|UNK| 
|:---|:---:|:---:|:---:|:---:|
|This Code|63.8|60.6|70.7|80.1|
|Paper|63.0|59.1|71.0|82.3|