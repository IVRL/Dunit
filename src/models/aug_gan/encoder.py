# -*- coding: utf-8 -*-
"""Encoder from the AugGAN paper"""
from torch.nn import LeakyReLU, Sequential

from ..base_module import BaseModule
from ..blocks import ConvBlock, ResBlock

class Encoder(BaseModule):
    """Encoder for AugGAN"""
    def __init__(self, nb_input_channels, dim, nb_blocks=3, **kwargs):
        super().__init__(**kwargs)
        self.layers = Sequential(
            ConvBlock(nb_input_channels, dim, kernel=7, padding=3,
                      non_linear=LeakyReLU, **kwargs),
            ConvBlock(dim, dim * 2, stride=2, non_linear=LeakyReLU),
            ConvBlock(dim * 2, dim * 4, stride=2, non_linear=LeakyReLU),
            *[ResBlock(dim * 4, **kwargs) for _ in range(nb_blocks)]
            )
