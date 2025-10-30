import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normalization(channels):
    return nn.InstanceNorm3d(channels, affine=True)


def get_nonlinearity():
    return nn.ReLU(inplace=True)


def conv_norm_nonl(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    # noinspection PyTypeChecker
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        get_normalization(out_channels),
        get_nonlinearity()
    )


def norm_nonl_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    # noinspection PyTypeChecker
    return nn.Sequential(
        get_normalization(in_channels),
        get_nonlinearity(),
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
    )


def convtr_norm_nonl(in_channels, out_channels, kernel_size=2, stride=2, padding=0):
    # noinspection PyTypeChecker
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=False),
        get_normalization(out_channels),
        get_nonlinearity()
    )


def norm_nonl_convtr(in_channels, out_channels, kernel_size=2, stride=2, padding=0):
    # noinspection PyTypeChecker
    return nn.Sequential(
        get_normalization(in_channels),
        get_nonlinearity(),
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=False),
    )


def down_block(in_channels, out_channels, kernel_size=3):
    # noinspection PyTypeChecker
    return nn.Sequential(
        norm_nonl_conv(in_channels, out_channels,
                       kernel_size=2, stride=2, padding=0),
        norm_nonl_conv(out_channels, out_channels,
                       kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    )


def up_block(in_channels, out_channels, kernel_size=3):
    # noinspection PyTypeChecker
    return nn.Sequential(
        norm_nonl_convtr(in_channels, out_channels,
                         kernel_size=2, stride=2, padding=0),
        norm_nonl_conv(out_channels, out_channels,
                       kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    )


class PreActResBlock(nn.Module):
    # Preactivation Residual Block
    def __init__(self, channels, kernel_size=3, num_convs=2):
        super().__init__()

        blocks = []
        for i in range(num_convs):
            blocks.append(norm_nonl_conv(channels, channels, kernel_size=kernel_size,
                                         stride=1, padding=kernel_size // 2))
        self.block = nn.Sequential(*blocks)

    def forward(self, inp):
        skip = inp
        out = self.block(inp)
        out += skip
        return out
