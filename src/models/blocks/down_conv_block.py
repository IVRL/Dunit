# -*- coding: utf-8 -*-
"""Module for down-convolution block"""
from torch.nn import Conv2d, Sequential, Identity

from ..base_module import BaseModule
from .mixins.conv_block import WithConvBlockMixin
from .mixins.padding import WithPaddingMixin

class DownConvBlock(WithPaddingMixin, WithConvBlockMixin, BaseModule):
    """Module implementing a modular DownConvBlock

    A DownConvBlock is a succession of Conv + Conv + norm + non-linear layers
    """
    def __init__(self, nb_input_channels, nb_output_channels,
                 nb_intermediary_output_channels=None,
                 non_linear2=Identity, **kwargs):
        if nb_intermediary_output_channels is None:
            nb_intermediary_output_channels = nb_output_channels
        super().__init__(nb_input_channels=nb_intermediary_output_channels,
                         nb_output_channels=nb_output_channels, **kwargs)
        self.layers = Sequential(
            self.padding_layer,
            Conv2d(nb_input_channels, nb_intermediary_output_channels,
                   kernel_size=kwargs.get("kernel", 3),
                   stride=kwargs.get("stride", 1)),
            non_linear2(),
            self.conv_block,
            )
