# -*- coding: utf-8 -*-
"""Mixins for convolutional blocks"""
from torch.nn import BatchNorm2d, ReLU, ZeroPad2d

from ..conv_block import ConvBlock

class WithConvBlockMixin():# pylint: disable=too-few-public-methods
    """Mixin for nets with a ConvBlock"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nb_input_channels = kwargs.get("nb_channels",
                                       kwargs.get("nb_input_channels"))
        nb_output_channels = kwargs.get("nb_channels",
                                        kwargs.get("nb_output_channels"))
        kernel = kwargs.get("kernel2", kwargs.get("kernel", 3))
        stride = kwargs.get("stride2", kwargs.get("stride", 1))
        padding = kwargs.get("padding2", kwargs.get("padding", 1))
        self.conv_block = ConvBlock(
            nb_input_channels,
            nb_output_channels,
            kernel=kernel,
            stride=stride,
            padding=padding,
            norm=kwargs.get("norm", BatchNorm2d),
            non_linear=kwargs.get("non_linear", ReLU),
            padding_module=kwargs.get("padding_module", ZeroPad2d))
