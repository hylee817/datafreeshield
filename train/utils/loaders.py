# from enum import Enum
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

def load_medmnist(subclass="tissuemnist"):
    test_transform = transforms.Compose([transforms.ToTensor()])
    target_transform = transforms.Lambda(lambda x: x[0],)
    DataClass = getattr(medmnist, INFO[subclass]['python_class']) #dataset of target	
    train_dataset = DataClass(split='train', transform=test_transform, target_transform=target_transform, download=True, as_rgb=True)
    test_dataset = DataClass(split='test', transform=test_transform, target_transform=target_transform, download=True, as_rgb=True)

    return train_dataset, test_dataset


def load_svhn(data_root="/datasets/svhn"):
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = None
    train_dataset = dsets.SVHN(root=data_root,
                    split='train', #validation data
                    transform=test_transform,
                    download=True)    
    test_dataset = dsets.SVHN(root=data_root,
                            split='test', #validation data
                            transform=test_transform,
                            download=True)
    return train_dataset, test_dataset
        

def load_cifar10(data_root="/datasets/cifar10"):
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = None
    train_dataset = dsets.CIFAR10(root=data_root,
                                    train=True,
                                    transform=test_transform)

    test_dataset = dsets.CIFAR10(root=data_root,
                                    train=False,
                                    transform=test_transform)
    return train_dataset, test_dataset


def load_cifar100(data_root="/datasets/cifar100"):
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = dsets.CIFAR100(root=data_root,
                                    train=True,
                                    transform=test_transform)
    test_dataset = dsets.CIFAR100(root=data_root,
                                    train=False,
                                    transform=test_transform)    
    return train_dataset, test_dataset
