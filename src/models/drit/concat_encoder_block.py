"""Block for ConcatStyleEncoder"""
from torch.nn import Sequential, LeakyReLU, AvgPool2d, ReflectionPad2d, \
    Identity, Conv2d

from ..base_module import BaseModule
from ..blocks import ConvBlock

class ConcatEncoderBlock(BaseModule):
    """Block for ConcatStyleEncoder"""
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__(kwargs)
        self.layers = Sequential(
            LeakyReLU(0.2),
            ConvBlock(input_channels, input_channels, kernel=3, stride=1,
                      padding_module=ReflectionPad2d, norm=Identity),
            LeakyReLU(0.2),
            ConvBlock(input_channels, output_channels, kernel=3, stride=1,
                      padding_module=ReflectionPad2d, norm=Identity),
            AvgPool2d(2, 2),)
        self.skip_layers = Sequential(
            AvgPool2d(2, 2),
            Conv2d(input_channels, output_channels, kernel_size=1))
        self.skip_connection = True
