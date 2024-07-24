from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from models import ResNet18, ResNet50
from tqdm import tqdm
import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data
import time
import random

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=76, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', default='resnet18',
                    help='model name')
parser.add_argument('--gpu',default=0, type=int)
parser.add_argument('--data_flag', default=None, type=str)
parser.add_argument('--as_rgb', action='store_true')
parser.add_argument('--exp_name', default=None, help="DB exp name")
parser.add_argument('--local', action='store_true')
parser.add_argument('--db', action='store_true')
args = parser.parse_args()

if not args.local:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

# settings
# model_dir = args.model_dir
model_dir = './save_{}/{}/'.format(args.data_flag, args.model)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

############
info = INFO[args.data_flag]
task = info['task']
n_channels = 3 #if args.as_rgb else info['n_channels']
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=True, as_rgb=args.as_rgb)
test_dataset = DataClass(split='test', transform=data_transform, download=True, as_rgb=args.as_rgb)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
# train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*args.batch_size, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*args.batch_size, shuffle=False)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.squeeze().long()
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            target = target.squeeze().long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target.squeeze().long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    if args.model == "resnet18":
        model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif args.model == "resnet50":
        model = ResNet50(in_channels=n_channels, num_classes=n_classes)
    model = model.to(device)
    print("model loaded")

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_best = 1000.0
    acc_best = 0.0
    epoch_best = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        loss_train, acc_train = eval_train(model, device, train_loader)
        print('================================================================')
        loss_test, acc_test = eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        # if epoch % args.save_freq == 0:
        if acc_test > acc_best:
            acc_best = acc_test
            epoch_best = epoch
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model.pt'))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-res-checkpoint.tar'))
            print("saved best model")

        torch.save(model.state_dict(), os.path.join(model_dir, 'model_last.pt'))

if __name__ == '__main__':
    main()
