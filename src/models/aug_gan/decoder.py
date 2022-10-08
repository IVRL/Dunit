# -*- coding: utf-8 -*-
"""Decoder for AugGAN"""
from torch.nn import Sequential, ConvTranspose2d, Conv2d, ReflectionPad2d, \
    BatchNorm2d, Tanh, ReLU

from ..base_module import BaseModule

class Decoder(BaseModule):
    """Decoder for AugGAN"""
    def __init__(self, input_nc, output_nc, **kwargs):
        super().__init__(**kwargs)
        c_layers = Sequential(
            ConvTranspose2d(input_nc, input_nc // 2, kernel_size=3,
                            stride=2, padding=1, output_padding=1),
            kwargs.get('norm', BatchNorm2d)(input_nc // 2),
            ConvTranspose2d(input_nc // 2, input_nc // 4, kernel_size=3,
                            stride=2, padding=1, output_padding=1),
            kwargs.get('norm', BatchNorm2d)(input_nc // 4)
        )
        ts_layers = [
            ReflectionPad2d(3),
            Conv2d(input_nc // 4, output_nc, kernel_size=7),
            ]
        if output_nc == 3:
            ts_layers += [
                Tanh()
            ]
        else:
            ts_layers += [
                ReLU(inplace=True),
                Conv2d(output_nc, output_nc, kernel_size=1, stride=1)
            ]
        self.layers = Sequential(c_layers, *ts_layers)
