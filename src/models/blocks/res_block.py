# -*- coding: utf-8 -*-
"""Module for ResBlock"""
from torch.nn import Conv2d, BatchNorm2d, Sequential

from ..base_module import BaseModule
from .mixins.conv_block import WithConvBlockMixin
from .mixins.padding import WithPaddingMixin

class ResBlock(WithPaddingMixin, WithConvBlockMixin, BaseModule):
    """Module implementing a modular ResBlock"""
    def __init__(self, nb_channels, **kwargs):
        super().__init__(nb_channels=nb_channels, **kwargs)
        self.layers = Sequential(
            self.conv_block,
            self.padding_layer,
            Conv2d(nb_channels, nb_channels,
                   kernel_size=kwargs.get("kernel", 3),
                   stride=kwargs.get("stride", 1)),
            kwargs.get("norm2", kwargs.get("norm", BatchNorm2d))(nb_channels)
            )
        self.skip_connection = True
