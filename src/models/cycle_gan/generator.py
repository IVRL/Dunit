"""Generator for CycleGAN"""
from torch.nn import Conv2d, Sequential, InstanceNorm2d, ReflectionPad2d

from ..base_module import BaseModule
from ..blocks import ResBlock, ConvBlock, DownConvBlock, UpConvBlock

class CycleGANGenerator(BaseModule):
    """Generator for CycleGAN"""
    def __init__(self, options):
        super().__init__(options=options)
        padding_module = ReflectionPad2d
        self.layers = Sequential(
            #down-convolutions
            Sequential(
                ConvBlock(3, 64, kernel=7, stride=1, padding=3,
                          norm=InstanceNorm2d, padding_module=padding_module),
                DownConvBlock(64, 128, kernel=3, stride=2, padding=1,
                              stride2=1, norm=InstanceNorm2d,
                              padding_module=padding_module),
                DownConvBlock(128, 256, kernel=3, stride=2, padding=1,
                              stride2=1, norm=InstanceNorm2d,
                              padding_module=padding_module),
                ),
            # ResBlocks
            Sequential(
                *[ResBlock(256, norm=InstanceNorm2d,
                           padding_module=padding_module) for _ in range(9)]
            ),
            # up-convolutions
            Sequential(
                UpConvBlock(256, 128, kernel=3, stride=2, padding=1,
                            stride2=1, norm=InstanceNorm2d,
                            padding_module=padding_module),
                UpConvBlock(128, 64, kernel=3, stride=2, padding=1,
                            stride2=1, norm=InstanceNorm2d,
                            padding_module=padding_module),
                padding_module(3),
                Conv2d(64, 3, kernel_size=7, stride=1)
                )
            )
