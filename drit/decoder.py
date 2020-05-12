"""Decoder for DRIT"""
import torch
from torch.nn import Sequential, ConvTranspose2d, ReLU, Tanh, Linear, LayerNorm

from ..base_module import BaseModule
from .decoder_block import DecoderBlock
from ..blocks import ConvBlock

class Decoder(BaseModule):
    """Decoder for DRIT"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dec1 = DecoderBlock(256, 256)
        self.dec2 = DecoderBlock(256, 256)
        self.dec3 = DecoderBlock(256, 256)
        self.dec4 = DecoderBlock(256, 256)

        self.dec5 = Sequential(
            ConvBlock(256, 128, kernel=3, stride=2, padding=1,
                      output_padding=1, norm=lambda nb_channels: LayerNorm(
                          (nb_channels, 108, 108)),
                      conv_layer=ConvTranspose2d, transposed=True),
            ConvBlock(128, 64, kernel=3, stride=2, padding=1,
                      output_padding=1, norm=lambda nb_channels: LayerNorm(
                          (nb_channels, 216, 216)),
                      conv_layer=ConvTranspose2d, transposed=True),
            ConvTranspose2d(64, kwargs.get("nb_channels_output", 3),
                            kernel_size=1, stride=1, padding=0),
            Tanh())

        self.mlp = Sequential(
            Linear(8 + kwargs.get("nb_domains", 3), 256),
            ReLU(inplace=True),
            Linear(256, 256),
            ReLU(inplace=True),
            Linear(256, 256 * 4))

    def forward(self, content, style, domains):# pylint: disable=arguments-differ
        style = self.mlp(torch.cat([style, domains], dim=1))
        style1, style2, style3, style4 = [
            t.contiguous() for t in torch.split(style, 256, dim=1)]
        out1 = self.dec1(content, style1)
        out2 = self.dec2(out1, style2)
        out3 = self.dec3(out2, style3)
        out4 = self.dec4(out3, style4)
        return self.dec5(out4)
