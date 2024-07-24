import torch
import torch.nn as nn
import torch.nn.functional as F


def PGD_generate(model, labels, alpha=2/255, eps=8/255, steps=10):
    """
    Overridden.
    """
    # images = images.clone().detach().cuda()
    act = model.x2.data
    act = act.clone().detach().cuda()
    act.requires_grad = True #considered leaf node
    labels = labels.clone().detach().cuda()

    loss = nn.CrossEntropyLoss()

    adv_act = act.clone().detach()

    # if self.random_start:
    #     # Starting at a uniformly random point
    #     adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
    #     adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_act.requires_grad = True
        # outputs = model(adv_images)
        outputs = model.features.final_pool(model.features.stage3(model.features.stage2(adv_act)))
        outputs = model.output(outputs.view(outputs.size(0),-1))

        # Calculate loss
        cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_act,
                                    retain_graph=False, create_graph=False)[0]

        adv_act = adv_act.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_act - act, min=-eps, max=eps)
        adv_act = torch.clamp(act + delta, min=0, max=1).detach()

    return adv_act, act, delta

def PGD_train(model, adv_act, labels, opt, teacher_out):

    model.train()
    loss = nn.KLDivLoss().cuda()
    # loss = nn.CrossEntropyLoss().cuda()

    adv_act = adv_act.clone().detach().cuda()
    labels = labels.clone().detach().cuda()

    outputs = model.features.final_pool(model.features.stage3(model.features.stage2(adv_act)))
    outputs = model.output(outputs.view(outputs.size(0),-1))

    cost = loss(F.log_softmax(outputs, dim=1), F.softmax(teacher_out, dim=1))

    opt.zero_grad()
    cost.backward()
    opt.step()