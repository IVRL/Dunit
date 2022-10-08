# -*- coding: utf-8 -*-
"""Multi-scale discriminator"""
import torch
from torch.nn import AvgPool2d, ModuleList, Sequential, Conv2d, Identity,\
    functional as F
from torch.autograd import Variable

from ..base_module import BaseModule
from ..blocks import ConvBlock

class MultiScaleDiscriminator(BaseModule):
    """Multi-scale discriminator"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = kwargs
        self.downsample = AvgPool2d(3, stride=2, padding=[1, 1],
                                    count_include_pad=False)
        self.cnns = ModuleList()
        for _ in range(self.options.nb_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.options.last_layer_size
        cnn_x = [
            ConvBlock(self.options.nb_channels, dim, kernel=4, stride=2,
                      padding=1, norm=Identity,
                      **{key: value
                         for key, value in self.params.items()
                         if key != "norm"})
            ]

        for _ in range(self.options.nb_layers - 1):
            cnn_x.append(
                ConvBlock(dim, dim * 2, kernel=4, stride=2, padding=1,
                          **self.params))
            dim *= 2
        cnn_x.append(Conv2d(dim, 1, 1, 1, 0))
        cnn_x = Sequential(*cnn_x)
        return cnn_x

    def forward(self, input_):
        outputs = []
        for model in self.cnns:
            outputs.append(model(input_))
            input_ = self.downsample(input_)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        """Compute the loss to train the discriminator"""
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for _, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.options.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.options.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).to(self.device),
                                requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).to(self.device),
                                requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0),
                                                          all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1),
                                                          all1))
            else:
                assert 0, f"Unsupported GAN type: {self.options.gan_type}"
        return loss

    def calc_gen_loss(self, input_fake):
        """Compute the loss to train the generator"""
        outs0 = self.forward(input_fake)
        loss = 0
        for _, (out0) in enumerate(outs0):
            if self.options.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.options.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).to(self.device),
                                requires_grad=False)
                loss += torch.mean(
                    F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, f"Unsupported GAN type: {self.options.gan_type}"
        return loss
