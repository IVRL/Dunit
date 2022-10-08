"""Multi-scale discriminator for DRIT"""
from torch.nn import Conv2d, LeakyReLU, Sequential, ReflectionPad2d, \
    InstanceNorm2d, Identity, AvgPool2d, utils, ModuleList

from ..base_module import BaseModule
from ..blocks import ConvBlock

class MultiScaleDiscriminator(BaseModule):
    """Multi-scale discriminator for DRIT"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.downsample = AvgPool2d(3, stride=2, padding=1,
                                    count_include_pad=False)

        spectral_norm = kwargs.get("spectral_norm", False)
        nb_layers = kwargs.get("nb_layers", 4)
        final_layer = Conv2d(
            64 * (2 ** (nb_layers - 1)), 1, kernel_size=1, stride=1, padding=0)
        if spectral_norm:
            final_layer = utils.spectral_norm(final_layer)

        self.discriminators = ModuleList(
            [Sequential(
                ConvBlock(
                    kwargs.get('input_dim', 3), 64, kernel=4, stride=2,
                    padding=1, padding_module=ReflectionPad2d,
                    spectral_norm=spectral_norm,
                    norm=kwargs.get('norm', Identity), non_linear=LeakyReLU),
                *[ConvBlock(
                    64 * (2 ** i), 64 * (2 ** (i + 1)), kernel=4, stride=2,
                    padding_module=ReflectionPad2d,
                    spectral_norm=spectral_norm,
                    norm=InstanceNorm2d, non_linear=LeakyReLU)
                  for i in range(nb_layers - 1)],
                final_layer)
             for _ in range(kwargs.get("nb_scales", 3))])

    def forward(self, input_):
        outputs = []
        for discriminator in self.discriminators:
            outputs.append(discriminator(input_))
            input_ = self.downsample(input_)
        return outputs
