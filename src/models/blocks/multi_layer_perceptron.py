# -*- coding: utf-8 -*-
"""Module for multi-layer perceptrons"""
from torch.nn import Sequential, Identity

from ..base_module import BaseModule
from .linear import LinearBlock

class MultiLayerPerceptron(BaseModule):
    """Multi-layer perceptron"""
    def __init__(self, input_dim, output_dim, dim, nb_layers, **kwargs):
        super().__init__(**kwargs)
        self.layers = Sequential(
            LinearBlock(input_dim, dim, **kwargs),
            *[LinearBlock(dim, dim, **kwargs) for _ in range(nb_layers - 2)],
            LinearBlock(dim, output_dim, non_linear=Identity, norm=Identity,
                        **{key: value
                           for key, value in kwargs.items()
                           if key not in ('non_linear', 'norm')})
            )

    def forward(self, input_):
        return super().forward(input_.view(input_.size(0), -1))
