import argparse
import os
import time
import traceback
import sys
import copy
import numpy as np
import torch
import csv

# models
from pytorchcv.model_provider import get_model as ptcv_get_model
from models import ResNet18, ResNet50

from autoattack import AutoAttack
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchattacks
import torchvision.datasets as dsets
from tqdm import tqdm

import medmnist
from medmnist import INFO, Evaluator

import MongoManager
import random


def _load_dataset(
        dataset,
        n_examples=None):
    batch_size = 100
    test_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        mu = torch.tensor(self.mean).view(3, 1, 1).cuda()
        std = torch.tensor(self.std).view(3, 1, 1).cuda()
        return (x - mu)/std


def main():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--dataset', type=str, choices=['svhn','cifar10','cifar100','mnist',
                                                        'tissuemnist','dermamnist','bloodmnist','octmnist',
                                                        'pathmnist','organamnist','organcmnist','organsmnist'])
    parser.add_argument('--model', type=str) 
    parser.add_argument('--eps', type=int) #4, 8
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--root', type=str) #path to model dir
    parser.add_argument('--csv_file', type=str, default=None)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--db', action='store_true')
    parser.add_argument('--dc', action='store_true')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--distance', type=str, choices=["Linf","L2"], default=None)
    parser.add_argument('--local', action='store_true')

    args = parser.parse_args()

    if not args.local:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device('cuda')

    model_name = args.model
    data_name = args.dataset
    ckpt = args.root
    eps = args.eps
    if args.distance == "L2": 
        step_size = 15
    else:
        step_size = args.eps/4

    #save to DB
    if args.db:
        MongoManager.config['project'] = "AutoAttack_{}".format(args.exp_name)
        ssh_connected = False
        while not ssh_connected:
            try:
                mongo = MongoManager.DBHandler()
                print("SSH Connection Successful")
                ssh_connected = True
            except:
                sleep_period = random.randrange(1,10)
                print(f"SSH Connection Error...Retrying after {sleep_period} secs")
                time.sleep(sleep_period)

        exp_id = "{}_{}_{}".format(ckpt, eps, args.normalize)
        db_dict = {'experimentID':exp_id,
                    'ckpt':ckpt[:-1],
                    'attack':"AutoAttack",
                    'distance':args.distance,
                    'eps':eps,
                    'model':model_name,
                    'dataset':data_name,
                    'normalize':args.normalize,
                    'pgd_acc': None,
                    'clean_acc':None,
                    'aa_acc':None}
        print(mongo.insert_item_one(db_dict))


    ##------------------------------ load dataset ----------------------------##
    if args.dc:
        test_data_root = "/scratch/{}".format(data_name)
    else:
        test_data_root = "/datasets/{}".format(data_name)

    transform_set = transforms.Compose([transforms.ToTensor()])
    if data_name == "cifar10":
        n_classes = 10
        test_dataset = dsets.CIFAR10(root=test_data_root,
                                    train=False, #validation data
                                    transform=transform_set,
                                    download=True)
    elif data_name == "svhn":
        n_classes = 10
        test_dataset = dsets.SVHN(root=test_data_root,
                                    split='test', #validation data
                                    transform=transform_set,
                                    download=True)
    elif data_name == "cifar100":
        n_classes = 100
        test_dataset = dsets.CIFAR100(root=test_data_root,
                                    train=False, #validation data
                                    transform=transform_set,
                                    download=True)
    else:
        info = INFO[data_name]
        test_transform = transforms.Compose([transforms.ToTensor()])
        target_transform = transforms.Lambda(lambda x: x[0],)
        DataClass = getattr(medmnist, info['python_class']) #dataset of target	
        test_dataset = DataClass(split='test', transform=test_transform, target_transform=target_transform, download=True, as_rgb=True)
        n_classes = len(info['label'])
                 
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                batch_size=200,
                                shuffle=False,
                                num_workers=8)
    print("dataset is ready")

    ##------------------------------ load model ----------------------------##
    if "mnist" in data_name: #medmnist
        if model_name == "resnet18":
            net = ResNet18(in_channels=3, num_classes=n_classes)
        elif model_name == "resnet50":
            net = ResNet50(in_channels=3, num_classes=n_classes)
    else:
        net = ptcv_get_model(model_name + "_" + data_name, pretrained=False)

    if args.normalize:
        if "mnist" in data_name:
            mean = [0.5, 0.5, 0.5] #pytorchcv models follow
            std = [0.5, 0.5, 0.5]          
        else:
            mean = [0.4914, 0.4822, 0.4465] #pytorchcv models follow
            std = [0.2023, 0.1994, 0.2010]
        model = torch.nn.Sequential(Normalize(mean=mean, std=std), net)
    else:
        model = net

    try: # trained with normalization layer
        model.load_state_dict(torch.load(ckpt + "robust_model.pt"))
    except:
        net.load_state_dict(torch.load(ckpt + "robust_model.pt"))
        model = torch.nn.Sequential(Normalize(mean=mean, std=std),net)

    model.to(device)
    model.eval()
    print("model is ready")


    ##------------------------------ evaluate PGD ----------------------------##
    if args.distance == "Linf":
        atk = torchattacks.PGD(model, eps=eps/255, alpha=step_size/255, steps=10)
    elif args.distance == "L2":
        atk = torchattacks.PGDL2(model, eps=eps/255, alpha=step_size/255, steps=10)

    tot_attack_correct = 0
    total = 0
    for image, label in tqdm(test_loader):
        image, label = image.cuda(), label.cuda()
        perturbed_image = atk(image, label)
        output_adv = model(perturbed_image)

        _, predicted = torch.max(output_adv.data, 1)
        total += label.size(0)
        tot_attack_correct += (predicted == label).sum()
        
    print('Robust accuracy: %.2f %%' % (100 * float(tot_attack_correct) / total))
    if args.db:
        acc = 100 * float(tot_attack_correct) / total
        update_done = False
        while not update_done:
            try:
                mongo.update_item_one({'experimentID': exp_id}, {"$set": {'pgd_acc': acc}})
                update_done = True
            except:
                sleep_period = random.randrange(1,10)
                print(f"SSH Connection Error...Retrying after {sleep_period} secs")
                time.sleep(sleep_period)
						
    ##------------------------------ evaluate AutoAttack ----------------------------##
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    adversary = AutoAttack(model,
                            norm=args.distance,
                            eps=(eps/255.),
                            version='standard',
                            n_classes=n_classes,
                            device=device,
                            log_path=None,)
    
    clean_x_test, clean_y_test =_load_dataset(test_dataset)
    x_adv, robust_accuracy_dict, robust_acc = adversary.run_standard_evaluation(clean_x_test,
                                                clean_y_test,
                                                bs=128,
                                                return_accuracy=True)
    clean_acc = robust_accuracy_dict['clean']

    if args.db:
        mongo.update_item_one({'experimentID': exp_id}, {"$set": {'clean_acc':clean_acc*100., 'aa_acc': robust_acc*100.}})
        mongo.close_connection()

    print("AA is done...")
    if args.csv_file != None:
        with open("./"+args.csv_file, "a+") as f:
            csvwriter = csv.writer(f, delimiter=',')
            csvwriter.writerow([args.root, args.dataset, args.model, str(args.eps), str(clean_acc*100.0), str(robust_acc*100.0)])


if __name__ == '__main__': 
	main()
