from functools import reduce


# PyTorch Library
import torch
import torch.nn as nn
import torch.nn.functional as F


def chain(input, layers):
    x = input
    for layer in layers:
        x = layer(x)
    return x


def isin(x, sets):
    return reduce(lambda l, r: l or r, map(lambda elem: x == elem, sets))


@torch.no_grad()
def apply_functions(layer, target_models, funcs):
    if isin(type(layer), target_models):
        for func in funcs:
            func(layer)


@torch.no_grad()
def init_conv(m):
    print(m)
    if isin(type(m), [nn.Conv1d, nn.Conv2d, nn.Conv3d]):
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.xavier_uniform_(m.bias.data)


@torch.no_grad()
def init_linear(m):
    print(m)
    if isin(type(m), [nn.Linear]):
        nn.init.xavier_uniform_(m.weight.data)


def downsample2D(in_channels,
                 out_channels,
                 kernel_size,
                 strides=2,
                 padding=0,
                 padding_mode='zeros',
                 dilation=1,
                 bias=True,
                 activation=nn.ELU(),
                 groups=1,
                 apply_batchnorm=True,
                 track_running_stats=True):
    conv = nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     stride=strides,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=bias,
                     padding_mode=padding_mode)

    if apply_batchnorm:
        batchnorm = nn.BatchNorm2d(out_channels,
                                   eps=1e-5,
                                   track_running_stats=track_running_stats)
        return nn.Sequential(conv,
                             activation,
                             batchnorm)
    else:
        return nn.Sequential(conv,
                             activation)


def upsample2D(in_channels,
               out_channels,
               kernel_size,
               strides=2,
               padding=0,
               padding_mode='zeros',
               dilation=1,
               bias=True,
               activation=nn.ELU(),
               groups=1,
               apply_batchnorm=True,
               track_running_stats=True,
               dropout=0.5,
               apply_dropout=False):
    conv = nn.ConvTranspose2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=strides,
                              output_padding=padding,
                              groups=groups,
                              bias=bias,
                              dilation=dilation,
                              padding_mode=padding_mode)

    if apply_batchnorm:
        batchnorm = nn.BatchNorm2d(out_channels,
                                   eps=1e-5,
                                   track_running_stats=track_running_stats)
    if apply_dropout:
        dropout = nn.Dropout2d(p=dropout)
        return nn.Sequential(conv,
                             batchnorm,
                             activation,
                             dropout)

    else:
        return nn.Sequential(conv,
                             activation,
                             batchnorm)


def downsample3D(in_channels,
                 out_channels,
                 kernel_size,
                 strides=2,
                 padding=0,
                 padding_mode='zeros',
                 dilation=1,
                 bias=True,
                 activation=nn.ELU(),
                 groups=1,
                 apply_batchnorm=True,
                 track_running_stats=True):
    conv = nn.Conv3d(in_channels,
                     out_channels,
                     kernel_size,
                     stride=strides,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=bias,
                     padding_mode=padding_mode)

    if apply_batchnorm:
        batchnorm = nn.BatchNorm3d(out_channels,
                                   eps=1e-5,
                                   track_running_stats=track_running_stats)
        return nn.Sequential(conv,
                             activation,
                             batchnorm)
    else:
        return nn.Sequential(conv,
                             activation)


def upsample3D(in_channels,
               out_channels,
               kernel_size,
               strides=2,
               padding=0,
               padding_mode='zeros',
               dilation=1,
               bias=True,
               activation=nn.ELU(),
               groups=1,
               apply_batchnorm=True,
               track_running_stats=True,
               dropout=0.5,
               apply_dropout=False):
    conv = nn.ConvTranspose3d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=strides,
                              output_padding=padding,
                              groups=groups,
                              bias=bias,
                              dilation=dilation,
                              padding_mode=padding_mode)

    if apply_batchnorm:
        batchnorm = nn.BatchNorm3d(out_channels,
                                   eps=1e-5,
                                   track_running_stats=track_running_stats)
    if apply_dropout:
        dropout = nn.Dropout3d(p=dropout)
        return nn.Sequential(conv,
                             activation,
                             batchnorm,
                             dropout)

    else:
        return nn.Sequential(conv,
                             activation,
                             batchnorm)


def LateralConnect2D(upsample2D,
                     downsample2D,
                     filters):
    channels = upsample2D.view(upsample2D.size(0), -1)
    result = torch.add(upsample2D, nn.Conv2d(
        channels, filters, 1, (1, 1))(downsample2D))

    result = nn.Conv2D(channels, filters, 1, (1, 1))(result)

    return result


def LateralConnect3D(upsample3D,
                     downsample3D,
                     filters):
    channels = upsample3D.view(upsample3D.size(0), -1)
    result = torch.add(upsample3D, nn.Conv3d(
        channels, filters, 1, (1, 1, 1))(downsample3D))

    result = nn.Conv3d(channels, filters, 1, (1, 1, 1))(result)

    return result
