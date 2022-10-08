# -*- coding: utf-8 -*-
"""Generator for CartoonGAN"""
from torch.nn import Conv2d, Sequential, BatchNorm2d, ZeroPad2d

from ..base_module import BaseModule
from ..blocks import ResBlock, ConvBlock, DownConvBlock, UpConvBlock

class CartoonGANGenerator(BaseModule):
    """Generator for CartoonGAN"""
    def __init__(self, norm=BatchNorm2d):
        super().__init__()
        self.layers = Sequential(
            #down-convolutions
            Sequential(
                ConvBlock(3, 64, kernel=7, stride=1, padding=3, norm=norm),
                DownConvBlock(64, 128, kernel=3, stride=2, padding=1,
                              stride2=1, norm=norm),
                DownConvBlock(128, 256, kernel=3, stride=2, padding=1,
                              stride2=1, norm=norm),
                ),
            # ResBlocks
            Sequential(
                *[ResBlock(256, norm=norm) for _ in range(8)]
            ),
            # up-convolutions
            Sequential(
                UpConvBlock(256, 128, kernel=3, stride=2, padding=1,
                            stride2=1, norm=norm),
                UpConvBlock(128, 64, kernel=3, stride=2, padding=1,
                            stride2=1, norm=norm),
                ZeroPad2d(3),
                Conv2d(64, 3, kernel_size=7, stride=1)
                )
            )
