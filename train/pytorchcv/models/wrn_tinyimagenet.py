"""
    WRN for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.
"""

__all__ = ['TINYWRN','wrn16_10_tinyimagenet', 'wrn28_10_tinyimagenet', 'wrn40_8_tinyimagenet']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import conv3x3
from .preresnet import PreResUnit, PreResActivation
import torch

class WRNConv(nn.Module):
    """
    WRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    activate : bool
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 activate):
        super(WRNConv, self).__init__()
        self.activate = activate

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.activate:
            x = self.activ(x)
        return x


def wrn_conv1x1(in_channels,
                out_channels,
                stride,
                activate):
    """
    1x1 version of the WRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    activate : bool
        Whether activate the convolution block.
    """
    return WRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        activate=activate)


def wrn_conv3x3(in_channels,
                out_channels,
                stride,
                activate):
    """
    3x3 version of the WRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    activate : bool
        Whether activate the convolution block.
    """
    return WRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        activate=activate)


class WRNBottleneck(nn.Module):
    """
    WRN bottleneck block for residual path in WRN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    width_factor : float
        Wide scale factor for width of layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 width_factor):
        super(WRNBottleneck, self).__init__()
        mid_channels = int(round(out_channels // 4 * width_factor))

        self.conv1 = wrn_conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=1,
            activate=True)
        self.conv2 = wrn_conv3x3(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            activate=True)
        self.conv3 = wrn_conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=1,
            activate=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class WRNUnit(nn.Module):
    """
    WRN unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    width_factor : float
        Wide scale factor for width of layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 width_factor):
        super(WRNUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = WRNBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            width_factor=width_factor)
        if self.resize_identity:
            self.identity_conv = wrn_conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activate=False)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class WRNInitBlock(nn.Module):
    """
    WRN specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(WRNInitBlock, self).__init__()
        self.conv = WRNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            activate=True)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class TINYWRN(nn.Module):
    """
    WRN model from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    width_factor : float
        Wide scale factor for width of layers.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (64, 64)
        Spatial size of the expected input image.
    num_classes : int, default 200
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 in_channels=3,
                 in_size=(64, 64),
                 num_classes=200):
        super(TINYWRN, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3(
            # stride=2, #added to match 32x32 input
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), PreResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=False,
                    conv1_stride=False))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("post_activ", PreResActivation(in_channels=in_channels))
        # self.features.add_module("final_pool", nn.AvgPool2d(
        #     kernel_size=8,
        #     stride=1))
        
        # self.gap = nn.AvgPool2d(kernel_size=8, stride=1)
        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        # print("before pool: ",x.shape)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        # print("before flatten: ",x.shape)
        # x = x.view(x.size(0), -1)
        # print("after flatten: ",x.shape)
        x = self.output(x)
        return x

def get_wrn_tinyimagenet(num_classes,
                  blocks,
                  width_factor,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create WRN model for Tiny-ImageNet with specific parameters.

    Parameters:
    ----------
    num_classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    width_factor : int
        Wide scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    assert ((blocks - 4) % 6 == 0)
    layers = [(blocks - 4) // 6] * 3
    channels_per_layers = [16, 32, 64]
    init_block_channels = 16

    channels = [[ci * width_factor] * li for (ci, li) in zip(channels_per_layers, layers)]
    print(layers)
    print(channels)
    net = TINYWRN(
        channels=channels,
        init_block_channels=init_block_channels,
        num_classes=num_classes,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net

def wrn16_10_tinyimagenet(num_classes=200, **kwargs):
    """
    WRN-50-2 model from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_wrn_tinyimagenet(num_classes=num_classes, blocks=16, width_factor=10, model_name="wrn16_10_tinyimagenet", **kwargs)


def wrn28_10_tinyimagenet(num_classes=200, **kwargs):
    """
    WRN-50-2 model from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_wrn_tinyimagenet(num_classes=num_classes, blocks=28, width_factor=10, model_name="wrn28_10_tinyimagenet", **kwargs)

def wrn40_8_tinyimagenet(num_classes=200, **kwargs):
    """
    WRN-50-2 model from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_wrn_tinyimagenet(num_classes=num_classes, blocks=40, width_factor=8, model_name="wrn40_8_tinyimagenet", **kwargs)



def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        (wrn16_10_tinyimagenet, 200),
        (wrn28_10_tinyimagenet, 200),
        (wrn40_8_tinyimagenet, 200),
    ]

    for model, num_classes in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != wrn16_10_tinyimagenet or weight_count == 36479194) #TODO: mod
        assert (model != wrn28_10_tinyimagenet or weight_count == 36479194)
        assert (model != wrn40_8_tinyimagenet or weight_count == 36479194)


        x = torch.randn(1, 3, 64, 64)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, num_classes))


if __name__ == "__main__":
    _test()
