"""Blocks for decoder"""
import torch
from torch.nn import ReflectionPad2d, InstanceNorm2d, Identity, Sequential

from ..base_module import BaseModule
from ..blocks import ConvBlock

class AuxBlock(BaseModule):
    """Auxiliary block for decoder"""
    def __init__(self, dim, dim_extra, **kwargs):
        super().__init__(**kwargs)
        self.conv = ConvBlock(
            dim, dim, kernel=3, stride=kwargs.get("stride", 1),
            padding_module=ReflectionPad2d,
            norm=InstanceNorm2d, non_linear=Identity)
        self.blk = Sequential(
            ConvBlock(dim + dim_extra, dim + dim_extra, kernel=1, stride=1,
                      padding=0, norm=Identity),
            ConvBlock(dim + dim_extra, dim, kernel=1, stride=1,
                      padding=0, norm=Identity))

    def forward(self, input_):# pylint: disable=arguments-differ
        input_, noise = input_
        return self.blk(torch.cat([self.conv(input_), noise], dim=1)), noise

class DecoderBlock(BaseModule):
    """Main block for decoder"""
    def __init__(self, dim, dim_extra, **kwargs):
        super().__init__(kwargs)
        self.layers = Sequential(
            AuxBlock(dim, dim_extra),
            AuxBlock(dim, dim_extra))

    def forward(self, input_, style):# pylint: disable=arguments-differ
        style_expand = style.view(style.size(0), style.size(1), 1, 1).expand(
            style.size(0), style.size(1), input_.size(2), input_.size(3))
        output, _ = self.layers((input_, style_expand))
        return input_ + output
