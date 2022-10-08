# -*- coding: utf-8 -*-
"""Multi-task generator for AugGAN"""
import torch
from torch.nn import Sequential, CosineSimilarity

from ..base_module import BaseModule
from ..blocks import ResBlock
from .decoder import Decoder

class MultiTaskGenerator(BaseModule):
    """Multi-task generator for AugGAN"""
    def __init__(self, input_nc, output_nc, parse_nc,
                 nb_blocks=6, **kwargs):
        super().__init__()
        self.shared = Sequential(*[
            ResBlock(input_nc, **kwargs)
            for _ in range(nb_blocks)
            ])
        self.decoders = [
            Decoder(input_nc, output_nc, **kwargs),# image
            Decoder(input_nc, parse_nc, **kwargs)# segmentation
            ]
        self.cosine_similarity = CosineSimilarity()

    def forward(self, input_):
        shared_representation = self.shared_x(input_)
        return tuple(
            decoder(shared_representation) for decoder in self.decoders)

    def get_weight_sharing_loss(self):
        """Compute soft weight-sharing loss"""
        return self.cosine_similarity(
            *[torch.stack([
                layer.weight
                for layer in (decoder.common_layers)
                if 'Conv' in layer.__class__.__name__ and \
                    hasattr(layer, 'weight')
                ])
              for decoder in self.decoders
              ])
