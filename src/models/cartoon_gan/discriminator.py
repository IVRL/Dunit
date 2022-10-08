# -*- coding: utf-8 -*-
"""Discriminator for CartoonGAN"""
from torch.nn import Conv2d, Sequential, Identity, LeakyReLU,\
                     BatchNorm2d

from ..base_module import BaseModule
from ..blocks import ConvBlock, DownConvBlock

class CartoonGANDiscriminator(BaseModule):
    """Discriminator for CartoonGAN"""
    def __init__(self, norm=BatchNorm2d):
        super().__init__()
        non_linear = lambda: LeakyReLU(0.2)
        self.layers = Sequential(
            ConvBlock(3, 64, kernel=3, stride=1, padding=1,
                      non_linear=non_linear, norm=Identity),
            DownConvBlock(64, 256,
                          nb_intermediary_output_channels=128,
                          kernel=3, stride=2, padding=1,
                          non_linear=non_linear, stride2=1,
                          norm=norm, non_linear2=non_linear),
            DownConvBlock(256, 1024,
                          nb_intermediary_output_channels=512,
                          kernel=3, stride=2, padding=1,
                          non_linear=non_linear, stride2=1,
                          norm=norm, non_linear2=non_linear),
            ConvBlock(1024, 1024, kernel=3, stride=1, padding=1,
                      non_linear=non_linear, norm=norm),
            Conv2d(1024, 1, kernel_size=3, stride=1, padding=1)
            )
