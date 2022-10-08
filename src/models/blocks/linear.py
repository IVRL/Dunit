# -*- coding: utf-8 -*-
"""Module for linear block"""
from torch.nn import Linear, BatchNorm2d, ReLU, Sequential

from ..base_module import BaseModule

class LinearBlock(BaseModule):
    """Module implementing a modular LinearBlock

    A LinearBlock is a succession of Linear + norm + non-linear layers
    """
    def __init__(self, nb_input_channels, nb_output_channels, **kwargs):
        super().__init__(**kwargs)
        self.layers = Sequential(
            Linear(nb_input_channels, nb_output_channels,
                   bias=kwargs.get("bias", True)),
            kwargs.get("norm", BatchNorm2d)(nb_output_channels),
            kwargs.get("non_linear", ReLU)(),
            )
