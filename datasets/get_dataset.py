from torchvision import transforms

from .mnist import *
from .svhn import *
from .usps import *

def get_dataset(task):
    if task == 's2m':
        train_dataset = SVHN('../data', split='train', download=True,
                    transform=transforms.Compose([
                       transforms.Resize(32),
                       transforms.ToTensor(),
                       transforms.Normalize((0.4376, 0.4437, 0.4728), (0.1980, 0.2010, 0.1970))
                   ]))
        
        test_dataset = MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
                ]))
    elif task == 'u2m':
        train_dataset = USPS('../data', train=True, download=True,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ]))
        
        test_dataset = MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    else:
        train_dataset = MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,))
                   ]))

        test_dataset = USPS('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    
    return relabel_dataset(train_dataset, test_dataset, task)

def relabel_dataset(train_dataset, test_dataset, task):
    image_path = []
    image_label = []
    if task == 's2m':
        for i in range(len(train_dataset.data)):
            if int(train_dataset.labels[i]) < 5:
                image_path.append(train_dataset.data[i])
                image_label.append(train_dataset.labels[i])
        train_dataset.data = image_path
        train_dataset.labels = image_label
    else:
        for i in range(len(train_dataset.train_data)):
            if int(train_dataset.train_labels[i]) < 5:
                image_path.append(train_dataset.train_data[i])
                image_label.append(train_dataset.train_labels[i])
        train_dataset.train_data = image_path
        train_dataset.train_labels = image_label

    for i in range(len(test_dataset.train_data)):
        if int(test_dataset.train_labels[i]) >= 5:
            test_dataset.train_labels[i] = 5
        
    return train_dataset, test_dataset