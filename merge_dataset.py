import os
import torch
import argparse

from torchvision import datasets, transforms
from DI_dataset import DeepInversionCIFAR10, DeepInversionSVHN

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str)
# parser.add_argument('--savename', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)


if __name__ == '__main__':
    args = parser.parse_args()
    
    prefix = f"dss_{args.dataset}_{args.model}"

    images = []
    targets = []
    xx = 0

    for sd in range(1,7):
        dset_path = os.path.join(args.root, prefix + f"_2000_id_{sd}")
        for j in range(4):
            for i in range(10):
                if i == 9:
                    path = os.path.join(dset_path, f'class_{i}_{j+1}.pt')
                else:
                    path = os.path.join(dset_path, f'class_{i}_{j}.pt')
                try:
                    xx += 1
                    x = torch.load(path)
                    images.append(x)
                    targets += [i] * x.shape[0]
                except:
                    print(path, f'not exist')

    images = torch.cat(images, dim=0)
    print(images.shape, len(targets), xx)

    if args.dataset == 'cifar10':
        dset = DeepInversionCIFAR10('/datasets/cifar10', data=images, targets=targets, transform=None)
    elif args.dataset == 'svhn':
        dset = DeepInversionSVHN('/datasets/cifar10', data=images, targets=targets, transform=None)
    else:
        raise Exception("Unsupported")

    save_pth = args.root + prefix + ".pt"
    print(save_pth)
    torch.save(dset, save_pth) #args.savename)
    print("saved!")