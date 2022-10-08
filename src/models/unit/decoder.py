# -*- coding: utf-8 -*-
"""Decoder for UNIT"""
from torch.nn import Sequential, Upsample, Tanh, ZeroPad2d, Identity, LayerNorm

from ..norms import AdaptiveInstanceNorm2d
from ..base_module import BaseModule
from ..blocks import ConvBlock, ResBlock

class Decoder(BaseModule):
    """Decoder for UNIT"""
    def __init__(self, n_upsample, nb_blocks, dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        # AdaIN residual blocks
        self.ada_in_resblocks = Sequential(
            *[ResBlock(dim,
                       norm=kwargs.get("res_norm", AdaptiveInstanceNorm2d),
                       **kwargs)
              for _ in range(nb_blocks)])
        # upsampling blocks
        self.upsampling = []
        features_dimension = 64
        for _ in range(n_upsample):
            features_dimension *= 2
            self.upsampling.extend([
                Upsample(scale_factor=2),
                ConvBlock(dim, dim // 2, kernel=5, stride=1, padding=2,
                          norm=lambda x, dim=features_dimension: LayerNorm(
                              (x, dim, dim)),
                          bias=True, **kwargs)])
            dim //= 2
        self.layers = Sequential(
            self.ada_in_resblocks,
            *self.upsampling,
            ConvBlock(dim, output_dim, kernel=7, stride=1, padding=3,
                      norm=Identity, non_linear=Tanh, bias=True,
                      padding_module=kwargs.get('padding_module', ZeroPad2d)
                      )
            )
