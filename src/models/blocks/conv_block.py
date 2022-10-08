"""Module for convolutional block"""
from torch.nn import Conv2d, BatchNorm2d, ReLU, Sequential, ConvTranspose2d
from torch.nn.utils import spectral_norm

from ..base_module import BaseModule
from .mixins.padding import WithPaddingMixin

class ConvBlock(WithPaddingMixin, BaseModule):
    """Module implementing a modular ConvBlock

    A ConvBlock is a succession of Conv + norm + non-linear layers
    """
    def __init__(self, nb_input_channels, nb_output_channels, **kwargs):
        super().__init__(**kwargs)
        transposed = kwargs.get("transposed", False)
        conv_module = kwargs.get(
            "conv_layer", ConvTranspose2d if transposed else Conv2d)
        if not transposed:
            conv_layer = conv_module(nb_input_channels, nb_output_channels,
                                     kernel_size=kwargs.get("kernel", 3),
                                     stride=kwargs.get("stride", 1),
                                     bias=kwargs.get("bias", True))
        else:
            conv_layer = conv_module(nb_input_channels, nb_output_channels,
                                     kernel_size=kwargs.get("kernel", 3),
                                     stride=kwargs.get("stride", 1),
                                     dilation=kwargs.get("dilation", 1),
                                     padding=(
                                         kwargs.get("padding", 1) +
                                         kwargs.get("dilation", 1) * (
                                             kwargs.get("kernel", 3) - 1)
                                         ),
                                     output_padding=kwargs.get(
                                         "output_padding", 1),
                                     bias=kwargs.get("bias", True))
        if kwargs.get("spectral_norm", False):
            conv_layer = spectral_norm(conv_layer)
        self.layers = Sequential(
            self.padding_layer,
            conv_layer,
            kwargs.get("norm", BatchNorm2d)(nb_output_channels),
            kwargs.get("non_linear", ReLU)(),
            )
