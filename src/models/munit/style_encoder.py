# -*- coding: utf-8 -*-
"""Style encoder for UNIT"""
from torch.nn import Sequential, AdaptiveAvgPool2d, Conv2d

from ..base_module import BaseModule
from ..blocks import ConvBlock

class StyleEncoder(BaseModule):
    """Style encoder for UNIT"""
    def __init__(self, n_downsample, input_dim, dim, style_dim, **kwargs):
        super().__init__()
        self.layers = Sequential(
            ConvBlock(input_dim, dim, kernel=7, stride=1, padding=3,
                      bias=True, **kwargs),
            ConvBlock(dim, dim * 2, kernel=4, stride=2, padding=1,
                      bias=True, **kwargs),
            ConvBlock(dim * 2, dim * 4, kernel=4, stride=2, padding=1,
                      bias=True, **kwargs),
            *[ConvBlock(dim * 4, dim * 4, kernel=4, stride=2, padding=1,
                        bias=True, **kwargs) for _ in range(n_downsample - 2)],
            AdaptiveAvgPool2d(1),
            Conv2d(dim * 4, style_dim, 1, 1, 0)
            )
        self.output_dim = dim
