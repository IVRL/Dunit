"""PatchGAN discriminator for AugGAN"""
from torch import nn

from ..base_module import BaseModule

class NLayerDiscriminator(BaseModule):
    """PatchGAN discriminator for AugGAN"""

    def __init__(self, input_nc, ndf=64, n_layers=3, **kwargs):
        super().__init__(**kwargs)

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for layer_index in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** layer_index, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=4, stride=2, padding=2),
                kwargs['norm'](ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=4, stride=1, padding=1),
            kwargs['norm'](ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=4, stride=1, padding=1)]
        self.layers = nn.Sequential(*sequence)
