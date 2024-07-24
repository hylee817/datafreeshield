import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def loss_dfshield(model,
                x_natural,
                y,
                teacher_outputs,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                gamma=1.0,
                temp=6.0,
                distance='l_inf',
                inner_max='kl',
                freeze_bn=False):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    # avoid KL being NaN at point (0,0)
    EPS_ = 1e-8

    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                if inner_max == 'ce':
                    loss = F.cross_entropy(model(x_adv), y)
                elif inner_max == 'kl':
                    loss = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1)+EPS_)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                if inner_max == 'ce':
                    loss = (-1) * F.cross_entropy(model(x_adv), y)
                elif inner_max == 'kl':
                    loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                            F.softmax(model(x_natural), dim=1)+EPS_)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        # x_adv = torch.clamp(x_adv, 0.0, 1.0)
        raise Exception("Please specify norm distance")
    
    if not freeze_bn: model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # zero gradient
    # optimizer.zero_grad()
    # for p in model.parameters():
    #     p.grad = None

    # calculate robust loss
    logits = model(x_natural)
    logits_adv = model(x_adv)

    alpha = temp * temp
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits, dim=1)+EPS_)
    loss_t = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(teacher_outputs, dim=1)+EPS_)

    loss = beta * loss_robust + gamma * loss_t
    return logits_adv, loss, loss_robust, loss_t