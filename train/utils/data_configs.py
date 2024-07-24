# from enum import Enum
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from utils.loaders import *

CIFAR_MEAN=[0.4914, 0.4822, 0.4465]
CIFAR_STD=[0.2023, 0.1994, 0.2010]  

IMAGENET_MEAN=[0.485, 0.456, 0.406]
IMAGENET_STD=[0.229, 0.224, 0.225]

MEDMNIST_MEAN=[0.5,0.5,0.5]
MEDMNIST_STD=[0.5,0.5,0.5]

SupportedDataset = ['svhn', 'cifar10','cifar100', 
                    'deep_inversion_svhn', 'deep_inversion_cifar10','deep_inversion_cifar100', 'deep_inversion_tinyimagenet',
                    'deep_inversion_tissuemnist', 'deep_inversion_bloodmnist', 'deep_inversion_dermamnist', 'deep_inversion_organcmnist'
                    ]

class Basecfg():
    def __init__(self) -> None:
        self.nClasses = 10
        self.mean = [0.5,0.5,0.5]
        self.std = [0.5,0.5,0.5]

class svhn_cfg(Basecfg):
    def __init__(self) -> None:
        super().__init__()
        self.alias = 'svhn'
        self.mean = CIFAR_MEAN
        self.std = CIFAR_STD
        self.load = load_svhn
        self.n_classes = 10
        self.metric = ['accuracy']

class cifar10_cfg(Basecfg):
    def __init__(self, ) -> None:
        super().__init__()
        self.alias = 'cifar10'
        self.mean = CIFAR_MEAN
        self.std = CIFAR_STD
        self.load = load_cifar10
        self.n_classes = 10
        self.metric = ['accuracy']

class cifar100_cfg(Basecfg):
    def __init__(self) -> None:
        super().__init__()
        self.alias = 'cifar100'
        self.mean = CIFAR_MEAN
        self.std = CIFAR_STD
        self.load = load_cifar100
        self.n_classes = 100
        self.metric = ['accuracy']

class medmnist_cfg(Basecfg):
    def __init__(self) -> None:
        super().__init__()
        self.alias = 'medmnist'
        self.mean = MEDMNIST_MEAN
        self.std = MEDMNIST_STD
        self.load = load_medmnist
        self.n_classes = None
        self.subclass = None
        self.info = None
        self.metric = ['acuracy', 'f1_score']
