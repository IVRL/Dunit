"""ConcatStyle encoder for DRIT"""
from torch.nn import Sequential, ReflectionPad2d, LeakyReLU, Identity, \
    AdaptiveAvgPool2d, Linear

from ..base_module import BaseModule
from ..blocks import ConvBlock
from .concat_encoder_block import ConcatEncoderBlock
from .utils import merge_input_and_domain

class ConcatStyleEncoder(BaseModule):
    """ConcatStyle encoder for DRIT"""
    def __init__(self, nb_input_channels=3, output_dim=8, nb_domains=3,
                 **kwargs):
        super().__init__(**kwargs)
        nb_blocks = 4
        self.layers = Sequential(
            ConvBlock(
                nb_input_channels + nb_domains, 64, kernel=4, stride=2,
                padding=1, padding_module=ReflectionPad2d, norm=Identity,
                non_linear=Identity),
            *[ConcatEncoderBlock(
                64 * min(4, block_index + 1), 64 * min(4, block_index + 2))
              for block_index in range(nb_blocks - 1)],
            LeakyReLU(0.2),
            AdaptiveAvgPool2d(1))
        self.fc_mean = Linear(64 * min(4, nb_blocks + 1), output_dim)
        self.fc_var = Linear(64 * min(4, nb_blocks + 1), output_dim)

    def forward(self, input_, domains):# pylint: disable=arguments-differ
        output = super().forward(
            merge_input_and_domain(input_, domains)
            ).view(input_.size(0), -1)
        return self.fc_mean(output), self.fc_var(output)
