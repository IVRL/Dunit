"""Content discriminator for DRIT"""
from torch.nn import Conv2d, Sequential, InstanceNorm2d, Identity, \
    ReflectionPad2d, LeakyReLU

from ..base_module import BaseModule
from ..blocks import ConvBlock

class ContentDiscriminator(BaseModule):
    """Content discriminator"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = Sequential(
            *[ConvBlock(
                256, 256, kernel=7, stride=2, padding_module=ReflectionPad2d,
                norm=InstanceNorm2d, non_linear=LeakyReLU) for _ in range(3)],
            ConvBlock(256, 256, kernel=4, stride=1, padding=0, norm=Identity),
            Conv2d(256, kwargs.get("nb_domains", 3), kernel_size=1,
                   stride=1, padding=0)
            )

    def forward(self, input_):
        class_prediction = super().forward(input_)
        return class_prediction.view(
            class_prediction.size(0), class_prediction.size(1))
