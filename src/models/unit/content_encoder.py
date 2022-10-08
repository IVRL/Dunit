# -*- coding: utf-8 -*-
"""Content encoder for UNIT"""
from torch.nn import Sequential

from ..base_module import BaseModule
from ..blocks import ConvBlock, ResBlock

class ContentEncoder(BaseModule):
    """Content encoder for UNIT"""
    def __init__(self, n_downsample, nb_blocks, input_dim, dim, **kwargs):
        super().__init__(**kwargs)
        initial_dim = dim
        # downsampling blocks
        self.downsampling = []
        for _ in range(n_downsample):
            self.downsampling.append(
                ConvBlock(dim, 2 * dim, kernel=4, stride=2,
                          padding=1, bias=True, **kwargs))
            dim *= 2
        # residual blocks
        self.res_blocks = Sequential(
            *[ResBlock(dim) for _ in range(nb_blocks)]
            )
        self.layers = Sequential(
            ConvBlock(input_dim, initial_dim, kernel=7, stride=1, padding=3,
                      bias=True, **kwargs),
            *self.downsampling,
            self.res_blocks)
        self.output_dim = dim
