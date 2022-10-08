"""Discriminator for DRIT"""
from torch.nn import Conv2d, LeakyReLU, Sequential, ReflectionPad2d, \
    InstanceNorm2d, Identity, AdaptiveAvgPool2d

from ..base_module import BaseModule
from ..blocks import ConvBlock

class Discriminator(BaseModule):
    """Discriminator for DRIT"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        spectral_norm = kwargs.get("spectral_norm", False)

        self.layers = Sequential(
            ConvBlock(
                kwargs.get('nb_input_channels', 3), 64,
                kernel=3, stride=2, padding=1,
                padding_module=ReflectionPad2d, spectral_norm=spectral_norm,
                norm=kwargs.get('norm', Identity), non_linear=LeakyReLU),
            *[ConvBlock(
                64 * (2 ** i), 64 * (2 ** (i + 1)), kernel=3, stride=2,
                padding_module=ReflectionPad2d, spectral_norm=spectral_norm,
                norm=InstanceNorm2d, non_linear=LeakyReLU) for i in range(4)],
            ConvBlock(64 * (2 ** 4), 64 * (2 ** 4), kernel=3, stride=2,
                      norm=Identity, spectral_norm=spectral_norm,
                      padding_module=ReflectionPad2d, non_linear=LeakyReLU))

        self.conv1 = Conv2d(64 * (2 ** 4), 1, kernel_size=1, stride=1,
                            padding=1, bias=False)
        self.conv2 = Sequential(
            Conv2d(
                64 * (2 ** 4), kwargs.get("nb_domains", 3),
                kernel_size=kwargs.get("image_size", 216) // (2 ** 6),
                bias=False),
            AdaptiveAvgPool2d(1))

    def forward(self, input_):
        output = super().forward(input_)
        class_prediction = self.conv2(output)
        return self.conv1(output), class_prediction.view(
            class_prediction.size(0), class_prediction.size(1))
