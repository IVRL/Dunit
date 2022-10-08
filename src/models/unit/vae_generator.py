# -*- coding: utf-8 -*-
"""VAE generator for UNIT"""
import torch
from torch.nn import InstanceNorm2d
from torch.autograd import Variable

from ..base_module import BaseModule
from .content_encoder import ContentEncoder
from .decoder import Decoder

class VAEGenerator(BaseModule):
    """VAE generator"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # content encoder
        self.encoder = ContentEncoder(
            self.options.nb_downsample, self.options.nb_blocks,
            self.options.nb_channels, self.options.last_layer_size,
            norm=InstanceNorm2d, **kwargs)
        self.decoder = Decoder(
            self.options.nb_downsample, self.options.nb_blocks,
            self.encoder.output_dim, self.options.nb_channels,
            res_norm=InstanceNorm2d, **kwargs)

    def forward(self, images):# pylint: disable=arguments-differ
        # This is a reduced VAE implementation where we assume the outputs are
        # multivariate Gaussian distribution with mean = hiddens and
        # std_dev = all ones.
        hiddens, noise = self.encode(images)
        if self.training:
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        """Encode the images"""
        hiddens = self.encoder(images)
        noise = Variable(torch.randn(hiddens.size())).to(self.device)
        return hiddens, noise

    def decode(self, hiddens):
        """Decode images"""
        images = self.decoder(hiddens)
        return images
