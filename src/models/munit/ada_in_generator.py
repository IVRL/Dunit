# -*- coding: utf-8 -*-
"""Generator with Adaptative Instance norm"""
from torch.nn import Identity, InstanceNorm2d

from ..base_module import BaseModule
from ..blocks import MultiLayerPerceptron
from ..norms import AdaptiveInstanceNorm2d
from ..unit.content_encoder import ContentEncoder
from ..unit.decoder import Decoder
from .style_encoder import StyleEncoder

class AdaINGenerator(BaseModule):
    """Generator with Adaptative Instance norm"""
    # AdaIN auto-encoder architecture
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # style encoder
        self.style_encoder = StyleEncoder(
            4, self.options.nb_channels, self.options.last_layer_size,
            self.options.style_dim, norm=Identity, **kwargs)

        # content encoder
        self.content_encoder = ContentEncoder(
            self.options.nb_downsample, self.options.nb_blocks,
            self.options.nb_channels, self.options.last_layer_size,
            norm=InstanceNorm2d, **kwargs)
        self.decoder = Decoder(
            self.options.nb_downsample, self.options.nb_blocks,
            self.content_encoder.output_dim, self.options.nb_channels,
            res_norm=AdaptiveInstanceNorm2d, **kwargs)

        # MLP to generate AdaIN parameters
        self.mlp = MultiLayerPerceptron(
            self.options.style_dim, self._get_num_adain_params(),
            self.options.mlp_dim, nb_layers=3, norm=Identity, **kwargs)

    def forward(self, images):# pylint: disable=arguments-differ
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        """Encode an image to its content and style codes"""
        style_fake = self.style_encoder(images)
        content = self.content_encoder(images)
        return content, style_fake

    def decode(self, content, style):
        """Decode content and style codes to an image"""
        adain_params = self.mlp(style)
        self._assign_adain_params(adain_params)
        images = self.decoder(content)
        return images

    def _assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model"""
        for module in self.decoder.modules():
            if module.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :module.num_features]
                std = adain_params[:, module.num_features:2*module.num_features]
                module.bias = mean.contiguous().view(-1)
                module.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*module.num_features:
                    adain_params = adain_params[:, 2*module.num_features:]

    def _get_num_adain_params(self):
        """Return the number of AdaIN parameters needed by the model"""
        num_adain_params = 0
        for module in self.decoder.modules():
            if module.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * module.num_features
        return num_adain_params
