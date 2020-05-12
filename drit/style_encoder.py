"""Style encoder for DRIT"""
from torch.nn import Sequential, ReflectionPad2d, Identity, AdaptiveAvgPool2d, \
    Conv2d

from ..base_module import BaseModule
from ..blocks import ConvBlock
from .utils import merge_input_and_domain

class StyleEncoder(BaseModule):
    """Style encoder for DRIT"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = Sequential(
            ConvBlock(
                kwargs.get('nb_channels_input', 3) + kwargs.get('nb_domains'),
                64, kernel=7, stride=1, padding=3,
                padding_module=ReflectionPad2d,
                norm=Identity),
            ConvBlock(64, 128, kernel=4, stride=2, padding=1,
                      padding_module=ReflectionPad2d, norm=Identity),
            ConvBlock(128, 256, kernel=4, stride=2, padding=1,
                      padding_module=ReflectionPad2d, norm=Identity),
            ConvBlock(256, 256, kernel=4, stride=2, padding=1,
                      padding_module=ReflectionPad2d, norm=Identity),
            ConvBlock(256, 256, kernel=4, stride=2, padding=1,
                      padding_module=ReflectionPad2d, norm=Identity),
            AdaptiveAvgPool2d(1),
            Conv2d(256, kwargs.get('nb_channels_output', 8), kernel_size=1,
                   stride=1, padding=0)
            )

    def forward(self, input_, domains):# pylint: disable=arguments-differ
        output = super().forward(merge_input_and_domain(input_, domains))
        return output.view(output.size(0), -1)
