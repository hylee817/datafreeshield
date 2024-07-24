import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torch.optim as optim
import copy

def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor

def save_images(images, targets, step): #b, c, h, w
    for id in range(images.shape[0]):
        class_id = targets[id].item()
        place_to_store = 'save_bss_real/img_{:05d}_class_{}_step_{}.jpg'.format(id, class_id, step)
        image_np = images[id].data.cpu().numpy().transpose((1,2,0))
        pil_image = Image.fromarray((image_np*255).astype(np.uint8))
        pil_image.save(place_to_store)

def pgd_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=1.0,
              steps=5, #interpolation steps
              distance='l_inf'):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    pgd_samples = []
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_ce = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            pgd_samples.append(copy.deepcopy(x_adv.data)) #deep copy image
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits = model(x_natural)
    nat_probs = F.softmax(logits, dim=1)

    logits_adv = model(x_adv)
    adv_probs = F.softmax(logits_adv, dim=1)
    true_probs = torch.gather(adv_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    perturb_vec = x_adv - x_natural #direction vector
    # bss_outputs = []
    loss_pgd = torch.zeros(batch_size).cuda()
    for i, pgd_sample in enumerate(pgd_samples):
        logits_pgd = model(pgd_sample)

        #visualize
        # image_tensor = denormalize(bss_sample)
        # save_images(image_tensor, y, i)

        # if adv is misclassified, larger penalty 
        loss_pgd += F.cross_entropy(logits_pgd,y)
    loss_pgd = torch.div(loss_pgd, len(pgd_samples))
    loss_pgd = (1.0 / batch_size) * torch.sum(
       loss_pgd * (1.0000001 - true_probs)
    )
    
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y) #boosted ce
    loss = loss_adv + float(beta) * loss_pgd

    return loss
    ####################################
    # logits = model(x_natural)

    # logits_adv = model(x_adv)

    # nat_probs = F.softmax(logits, dim=1)

    # adv_probs = F.softmax(logits_adv, dim=1)

    # tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    # new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    # #boosted ce
    # loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    # true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
    # # print(kl(torch.log(adv_probs + 1e-12), nat_probs).shape) [200,10]
    # loss_robust = (1.0 / batch_size) * torch.sum(
    #     torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    # loss = loss_adv + float(beta) * loss_robust

    # return loss
