from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lmp
import lmp.nn as lnn
import lmp.nn.functional as LF
from lmp.util.typing import *


def get_normalization2d(channels):
    return nn.InstanceNorm2d(channels, affine=True)


def get_nonlinearity():
    return nn.ReLU(inplace=True)


def conv_norm_nonl2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    # noinspection PyTypeChecker
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=bias),
        get_normalization2d(out_channels),
        get_nonlinearity()
    )


def norm_nonl_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    # noinspection PyTypeChecker
    return nn.Sequential(
        get_normalization2d(in_channels),
        get_nonlinearity(),
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=bias),
    )


def convtr_norm_nonl2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True):
    # noinspection PyTypeChecker
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=bias),
        get_normalization2d(out_channels),
        get_nonlinearity()
    )


def norm_nonl_convtr2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=True):
    # noinspection PyTypeChecker
    return nn.Sequential(
        get_normalization2d(in_channels),
        get_nonlinearity(),
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=bias),
    )


def down_block2d(in_channels, out_channels, kernel_size=3, bias=True):
    # noinspection PyTypeChecker
    return nn.Sequential(
        norm_nonl_conv2d(in_channels, out_channels,
                         kernel_size=2, stride=2, padding=0, bias=bias),
        norm_nonl_conv2d(out_channels, out_channels,
                         kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=bias)
    )


def up_block2d(in_channels, out_channels, kernel_size=3, bias=True):
    # noinspection PyTypeChecker
    return nn.Sequential(
        norm_nonl_convtr2d(in_channels, out_channels,
                           kernel_size=2, stride=2, padding=0, bias=bias),
        norm_nonl_conv2d(out_channels, out_channels,
                         kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=bias)
    )


class PreActResBlock2d(nn.Module):
    # Preactivation Residual Block
    def __init__(self, channels, kernel_size=3, num_convs=2, bias=True):
        super().__init__()

        blocks = []
        for i in range(num_convs):
            blocks.append(norm_nonl_conv2d(channels, channels, kernel_size=kernel_size,
                                           stride=1, padding=kernel_size // 2, bias=bias))
        self.block = nn.Sequential(*blocks)

    def forward(self, inp):
        skip = inp
        out = self.block(inp)
        out += skip
        return out
