"""ConcatDecoder for DRIT"""
import torch
from torch.nn import Sequential, ConvTranspose2d, LayerNorm, Tanh, \
    InstanceNorm2d

from ..base_module import BaseModule
from ..blocks import ResBlock, ConvBlock
from .utils import merge_input_and_domain

class ConcatDecoder(BaseModule):
    """Decoder for DRIT"""
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        nb_channels_noise = 8
        nb_channels_input = 256

        self.shared_layers = kwargs.get(
            "decoder_shared_layers", ResBlock(
                nb_channels_input, norm=InstanceNorm2d))

        nb_input_channels = nb_channels_input + nb_channels_noise + \
            kwargs.get("nb_domains", 3)
        self.decoder_steps = [Sequential(
            *[ResBlock(nb_input_channels, norm=InstanceNorm2d)
              for _ in range(3)])]

        for _ in range(2):
            nb_output_channels = nb_input_channels // 2
            self.decoder_steps.append(ConvBlock(
                nb_input_channels, nb_output_channels,
                conv_layer=ConvTranspose2d, transposed=True,
                stride=2, norm=LayerNorm))
            nb_input_channels = nb_output_channels + nb_channels_noise

        nb_output_channels = nb_input_channels // 2
        self.decoder_steps.append(Sequential(
            ConvTranspose2d(nb_input_channels, 3,
                            kernel_size=1, stride=1),
            Tanh()))

    def forward(self, content, style, domains):# pylint: disable=arguments-differ
        content = self.shared_layers(content)
        content = merge_input_and_domain(content, domains)
        for step in self.decoder_steps:
            style_img = style.view(
                style.size(0), style.size(1), 1, 1).expand(
                    style.size(0), style.size(1),
                    content.size(2), content.size(3))
            content = step(torch.cat([content, style_img], dim=1))
        return content
