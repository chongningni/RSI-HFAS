import math

import torch
import torch.nn as nn
from collections import OrderedDict
import sys
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
        
        
        
class ResidualDenseBlock_8C(nn.Module):
    '''
    Residual Dense Block
    style: 8 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    '''

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero',norm_type=None, act_type='relu', mode='CNA'):
        super(ResidualDenseBlock_8C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = ConvBlock(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = ConvBlock(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = ConvBlock(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = ConvBlock(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv5 = ConvBlock(nc+4*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv6 = ConvBlock(nc+5*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv7 = ConvBlock(nc+6*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv8 = ConvBlock(nc+7*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv9 = ConvBlock(nc+8*gc, nc, 1, stride, bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x6 = self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1))
        x7 = self.conv7(torch.cat((x, x1, x2, x3, x4, x5, x6), 1))
        x8 = self.conv8(torch.cat((x, x1, x2, x3, x4, x5, x6, x7), 1))
        x9 = self.conv9(torch.cat((x, x1, x2, x3, x4, x5, x6, x7, x8), 1))
        return x9.mul(0.2) + x

    
    
    
    

def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,\
              act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, conv)


def DeconvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, padding=0, \
                act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, deconv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, deconv)



def get_valid_padding(kernel_size, dilation):
    """
    Padding value to remain feature size.
    """
    kernel_size = kernel_size + (kernel_size-1)*(dilation-1)
    padding = (kernel_size-1) // 2
    return padding

def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None

    layer = None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('[ERROR] Padding layer [%s] is not implemented!'%pad_type)
    return layer

def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = None
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!'%act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    layer = None
    if norm_type =='bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] Normalization layer [%s] is not implemented!'%norm_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict'%sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)
