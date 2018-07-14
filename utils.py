import torch
import shutil
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn as nn
import itertools
import os.path

from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler
from torchvision import datasets

def relabel_dataset(svhn_dataset, mnist_dataset):
    image_path = []
    image_label = []
    for i in range(len(svhn_dataset.data)):
        if int(svhn_dataset.labels[i]) < 5:
            image_path.append(svhn_dataset.data[i])
            image_label.append(svhn_dataset.labels[i])
    svhn_dataset.data = image_path
    svhn_dataset.labels = image_label

    for i in range(len(mnist_dataset.train_data)):
        if int(mnist_dataset.train_labels[i]) >= 5:
            mnist_dataset.train_labels[i] = 5
        
    return svhn_dataset, mnist_dataset


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def cal_acc(gt_list, predict_list, num):
    acc_sum = 0
    for n in range(num):
        y = []
        pred_y = []
        for i in range(len(gt_list)):
            gt = gt_list[i]
            predict = predict_list[i]
            if gt == n:
                y.append(gt)
                pred_y.append(predict)
        print ('{}: {:4f}'.format(n if n != 5 else 'unk', accuracy_score(y, pred_y)))
        acc_sum += accuracy_score(y, pred_y)
    print ('Avg Acc: {:4f}'.format(acc_sum / num))
    print ('Overall Acc : {:4f}'.format(accuracy_score(gt_list, predict_list)))

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))