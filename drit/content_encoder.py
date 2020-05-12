"""Content encoder for DRIT"""
from torch.nn import Sequential, ReflectionPad2d, LeakyReLU, Identity, \
    InstanceNorm2d

from ..base_module import BaseModule
from ..blocks import ResBlock, ConvBlock
from .gaussian_noise import GaussianNoiseLayer

class ContentEncoder(BaseModule):
    """Content encoder for DRIT"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shared_layers = kwargs.get("content_shared_layers", Sequential(
            ResBlock(256, norm=InstanceNorm2d, padding_module=ReflectionPad2d),
            GaussianNoiseLayer()
            ))

        self.layers = Sequential(
            ConvBlock(
                kwargs.get('input_dim', 3), 64, kernel=7, stride=1, padding=3,
                padding_module=ReflectionPad2d,
                norm=Identity, non_linear=LeakyReLU),
            ConvBlock(64, 128, kernel=3, stride=2, padding=1,
                      padding_module=ReflectionPad2d, norm=InstanceNorm2d),
            ConvBlock(128, 256, kernel=3, stride=2, padding=1,
                      padding_module=ReflectionPad2d, norm=InstanceNorm2d),
            *[ResBlock(256, norm=InstanceNorm2d, padding_module=ReflectionPad2d)
              for _ in range(3)],
            self.shared_layers)
