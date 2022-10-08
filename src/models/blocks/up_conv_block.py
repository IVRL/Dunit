# -*- coding: utf-8 -*-
"""Module for up-convolution block"""
from torch.nn import ConvTranspose2d, Sequential

from ..base_module import BaseModule
from .mixins.conv_block import WithConvBlockMixin
from .mixins.padding import WithPaddingMixin

class UpConvBlock(WithPaddingMixin, WithConvBlockMixin, BaseModule):
    """Module implementing a modular UpConvBlock

    A UpConvBlock is a succession of ConvTranspose + Conv + norm +
    non-linear layers
    """
    def __init__(self, nb_input_channels, nb_output_channels,
                 output_padding=1, dilation=1, **kwargs):
        super().__init__(nb_channels=nb_output_channels, **kwargs)
        self.layers = Sequential(
            self.padding_layer,
            ConvTranspose2d(nb_input_channels, nb_output_channels,
                            kernel_size=kwargs.get("kernel", 3),
                            stride=kwargs.get("stride", 1),
                            dilation=dilation,
                            padding=(
                                kwargs.get("padding", 1) +
                                dilation * (kwargs.get("kernel", 3) - 1)
                                ),
                            output_padding=output_padding
                           ),
            self.conv_block,
            )
