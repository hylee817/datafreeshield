'''
ResNet model inversion for CIFAR10.

Copyright (C) 2020 NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License (1-Way Commercial). To view a copy of this license, visit https://github.com/NVlabs/DeepInversion/blob/master/LICENSE
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy

import argparse
from pickle import TRUE
import random
import torch
import torch.nn as nn
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.transforms as transforms

import numpy as np
import os
import glob
import collections
from tqdm import tqdm
from PIL import Image
from torch.autograd import Variable

import medmnist
from medmnist import INFO, Evaluator
from models import ResNet18, ResNet50

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp, optimizers
    USE_APEX = True
except ImportError:
    print("Please install apex from https://www.github.com/nvidia/apex to run this example.")
    print("will attempt to run without it")
    USE_APEX = False

#provide intermeiate information
debug_output = False
debug_output = True


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''    
    if use_fp16:
        mean = np.array([0.5,0.5,0.5], dtype=np.float16)
        std = np.array([0.5,0.5,0.5], dtype=np.float16)
    else:
        mean = np.array([0.5,0.5,0.5])
        std = np.array([0.5,0.5,0.5])   

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor

def normalize(X, mode="default"):
    if mode == "cifar10":
        mean = [0.4914, 0.4822, 0.4465] #pytorchcv models follow
        stdev = [0.2023, 0.1994, 0.2010]
    elif mode == "svhn":
        mean = [0.4914, 0.4822, 0.4465]
        stdev = [0.2023, 0.1994, 0.2010]  
    elif mode == "default":
        mean = [0.5, 0.5, 0.5]
        stdev = [0.5, 0.5, 0.5]
    else:
        raise Exception("Unsupported normalization mode")
    
    mu = torch.tensor(mean).view(3, 1, 1).cuda()
    std = torch.tensor(stdev).view(3, 1, 1).cuda()
    return (X - mu)/std

def get_images(net, num_total_images=50000, bs=256, epochs=1000, idx=-1, var_scale=0.00005,
               net_student=None, prefix=None, train_writer = None, global_iteration=None,
               use_amp=False, target_coeff=1.0,
               bn_reg_scale = 0.0, random_labels = False, seed=0, save_root=None,  rand_mode=None, shift_scale=0.0):
    '''
    Function returns inverted images from the pretrained model, parameters are tight to CIFAR dataset
    args in:
        net: network to be inverted
        bs: batch size
        epochs: total number of iterations to generate inverted images, training longer helps a lot!
        idx: an external flag for printing purposes: only print in the first round, set as -1 to disable
        var_scale: the scaling factor for variance loss regularization. this may vary depending on bs
            larger - more blurred but less noise
        net_student: model to be used for Adaptive DeepInversion
        prefix: defines the path to store images
        competitive_scale: coefficient for Adaptive DeepInversion
        train_writer: tensorboardX object to store intermediate losses
        global_iteration: indexer to be used for tensorboard
        use_amp: boolean to indicate usage of APEX AMP for FP16 calculations - twice faster and less memory on TensorCores
        optimizer: potimizer to be used for model inversion
        inputs: data place holder for optimization, will be reinitialized to noise
        bn_reg_scale: weight for r_feature_regularization
        random_labels: sample labels from random distribution or use columns of the same class
        l2_coeff: coefficient for L2 loss on input
    return:
        A tensor on GPU with shape (bs, 3, 32, 32) for CIFAR
    '''

    orig_target_coeff = target_coeff
    orig_bn_reg_scale = bn_reg_scale

    # set up criteria for optimization
    criterion = nn.CrossEntropyLoss()
    # kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
    log_soft = nn.LogSoftmax(dim=1).cuda()
    counter = [seed * num_total_images]*n_classes

    # place holder for inputs
    data_type = torch.half if args.amp else torch.float

    num_samples_per_iteration = bs
    class_id = -1
    num_samples_left_to_generate = num_total_images

    origin_net = copy.deepcopy(net)

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    iter_num = -1
    while num_samples_left_to_generate > 0:

        iter_num += 1 
        best_cost = 1e6
        class_id = (class_id + 1) % n_classes
        num_samples_left_to_generate -= num_samples_per_iteration

        net = copy.deepcopy(origin_net)

        # initialize gaussian inputs
        inputs = torch.randn((num_samples_per_iteration, 3, 28, 28), requires_grad=True, device='cuda', dtype=data_type)
        optimizer = optim.Adam([inputs], lr=args.di_lr)
 
        # inputs.data = torch.randn((num_samples_per_iteration, 3, 32, 32), requires_grad=True, device='cuda')
 
        optimizer.state = collections.defaultdict(dict)  # Reset state of optimizer
 
        # target outputs to generate
        if random_labels:
            targets = torch.LongTensor([random.randint(0,n_classes-1) for _ in range(bs)]).cuda()
        else:
            # targets = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 25 + [0, 1, 2, 3, 4, 5]).to('cuda')
            targets = torch.LongTensor([class_id] * num_samples_per_iteration).cuda()
            # for computing point-wise ce
            one_hot = Variable(torch.zeros(targets.size()[0], n_classes)).cuda()
            one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        ## Create hooks for feature statistics catching
        loss_r_feature_layers = []
        for module in net.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))
 
        # setting up the range for jitter
        lim_0, lim_1 = 2, 2

        if rand_mode == 'dss':
            target_coeff = orig_target_coeff * (torch.rand(1).cuda() + shift_scale) 
            bn_reg_scale = orig_bn_reg_scale * (torch.rand(1).cuda() + shift_scale)
            print("COEFF USED: CE:{:.4f} Feature(BNS):{:.4f}, TV(smoothing):{:.6f} ".format(target_coeff.item(), bn_reg_scale.item(), var_scale))
        elif rand_mode == 'linear':
            target_coeff = torch.rand(1).cuda() + shift_scale 
            bn_reg_scale = (1.0 + shift_scale) - target_coeff 
            print("COEFF USED: CE:{:.4f} Feature(BNS):{:.4f}, TV(smoothing):{:.6f} ".format(target_coeff.item(), bn_reg_scale.item(), var_scale))

        for epoch in tqdm(range(epochs)):
            # apply random jitter offsets
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            inputs_jit = torch.roll(inputs, shifts=(off1,off2), dims=(2,3))
 
            # foward with jit images
            optimizer.zero_grad()
            net.zero_grad()
            outputs = net(inputs_jit)
            if 'ce_pointwise' in rand_mode:
                ce_pointwise = (-(one_hot * log_soft(outputs)).sum(dim=1))
                loss = (target_coeff * ce_pointwise).mean()
            else:
                loss = target_coeff * criterion(outputs, targets)
            loss_target = loss.item()       
 
            # apply total variation regularization
            diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
            diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
            diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
            diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
            loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
            loss = loss + var_scale*loss_var
 
            # R_feature loss
            # print([mod.r_feature.item() for mod in loss_r_feature_layers])
            loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
            loss = loss + bn_reg_scale*loss_distr # best for noise before BN
 
            if debug_output and epoch % 200==0:
                tqdm.write(f"It {epoch}\t Losses: total: {loss.item():3.3f},\ttarget: {loss_target:3.3f} \tR_feature_loss unscaled:\t {loss_distr.item():3.3f}")
                vutils.save_image(inputs.data.clone(),
                                  './{}/output_{}.png'.format(prefix, epoch//200),
                                  normalize=True, scale_each=True, nrow=10)
 
            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data
 
            loss.backward()
 
            optimizer.step()
 
        outputs=net(best_inputs)
        _, predicted_teach = outputs.max(1)
 
 
        if idx == 0:
            print('Teacher correct out of {}: {}, loss at {}'.format(bs, predicted_teach.eq(targets).sum().item(), criterion(outputs, targets).item()))
 
        name_use = "best_images"
        if prefix is not None:
            name_use = prefix + name_use
        next_batch = len(glob.glob("./%s/*.png" % name_use)) // 1
 
        #save grid
        vutils.save_image(best_inputs[:20],
                          './{}/output_{}.png'.format(name_use, next_batch),
                          normalize=True, scale_each = True, nrow=10) #normalize: If True, shift the image to the range (0, 1),by the min and max values specified by ``value_range``. Default: ``False``.
        
        # save as images 0-255
        best_inputs = denormalize(best_inputs.detach().cpu())
        # save as file
        filename = f'random_idx_{iter_num}.pt'
        # filename = f'class_{class_id}_{(num_total_images - num_samples_left_to_generate) // (num_samples_per_iteration * n_classes)}.pt'
        torch.save(best_inputs.detach().cpu(), f'{os.path.join(save_root, filename)}')
        torch.save(targets.detach().cpu(), f'{os.path.join(save_root, filename.replace("random_idx", "labels"))}')

        del net

def save_images(images, class_id, save_root, counter):
    # method to store generated images locally
    # local_rank = torch.cuda.current_device()
    # for id in range(images.shape[0]):
    #     class_id = targets[id].item()
    save_dir = os.path.join(save_root, str(class_id))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for img in images:
        save_name = save_dir + '/class{}_img_{}.png'.format(class_id, counter[class_id])
        image_np = img.data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(save_name)
        counter[class_id] += 1
        
    print("Image saved at: {}".format(save_dir))
    print(counter)

# def test():f
#     print('==> Teacher validation')
#     net_teacher.eval()
#     test_loss = 0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.cuda(), targets.cuda() #.to(device), targets.to(device)
#             outputs = net_teacher(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#     print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
#           % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 DeepInversion')
    parser.add_argument('--num_total_images', default=50000, type=int, help='# of total images to generate')
    parser.add_argument('--bs', default=250, type=int, help='batch size')
    parser.add_argument('--iters_mi', default=2000, type=int, help='number of iterations for model inversion')
    parser.add_argument('--di_lr', default=0.1, type=float, help='lr for deep inversion')
    parser.add_argument('--di_var_scale', default=2.5e-5, type=float, help='TV L2 regularization coefficient')
    # parser.add_argument('--di_l2_scale', default=0.0, type=float, help='L2 regularization coefficient')
    parser.add_argument('--r_feature_weight', default=1.0, type=float, help='weight for BN regularization statistic')
    parser.add_argument('--amp', action='store_true', help='use APEX AMP O1 acceleration')
    parser.add_argument('--target_scale', default=1.0, type=float, help='Cross Entropy Loss Coefficient')
    
    parser.add_argument('--exp_descr', default="try1", type=str, help='name to be added to experiment name')
    parser.add_argument('--model', default="resnet18", type=str, help='teacher model\'s name')
    parser.add_argument('--seed', default=0, type=int, help='manual seed')
    parser.add_argument('--save_root', required=True, type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_flag', type=str, default=None)

    parser.add_argument('--rand_mode', choices=['dss', 'linear'], default=None)
    parser.add_argument('--shift_scale', type=float, default=0.0)

    args = parser.parse_args()

    # print(f"loading {args.model}")

    torch.manual_seed(args.seed)
    
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    info = INFO[args.data_flag]
    n_channels = 3#self.info['n_channels'] #pretrained architecture
    n_classes = len(info['label'])

    ckpt = torch.load(f"../pretrained/{args.data_flag}/model.pt")
    if args.model == "resnet18":
        net_teacher = ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif args.model == "resnet50":
        net_teacher = ResNet50(in_channels=n_channels, num_classes=n_classes)
    net_teacher.load_state_dict(ckpt)
    net_teacher = net_teacher.cuda()

    net_student = None

    criterion = nn.CrossEntropyLoss()

    net_teacher.eval() #important, otherwise generated images will be non natural

    cudnn.benchmark = True


    batch_idx = 0
    prefix = "runs/data_generation/"+args.exp_descr+"/"

    for create_folder in [prefix, prefix+"/best_images/"]:
        if not os.path.exists(create_folder):
            os.makedirs(create_folder)

    train_writer = None  # tensorboard writter
    global_iteration = 0

    print("Starting model inversion")
    save_root = os.path.join(args.save_root, args.exp_descr)
    print("Result images will be saved at:{}".format(save_root))

    inputs = get_images(net=net_teacher, num_total_images=args.num_total_images, bs=args.bs, epochs=args.iters_mi, idx=batch_idx,
                        net_student=net_student, prefix=prefix,
                        train_writer=train_writer, global_iteration=global_iteration, use_amp=args.amp,
                        bn_reg_scale=args.r_feature_weight, var_scale=args.di_var_scale, target_coeff=args.target_scale,
                        random_labels=True, seed=args.seed,save_root=save_root, rand_mode=args.rand_mode, shift_scale=args.shift_scale)