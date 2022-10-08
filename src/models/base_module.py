# -*- coding: utf-8 -*-
"""Extension of torch.nn.Module for our own usage

Avoid calls to __init__ with wrong arguments
"""
from torch.nn import Module, Sequential, Identity

from ..with_device import WithDeviceMixin

class BaseModule(WithDeviceMixin, Module):
    """Base class for all modules"""

    def __init__(self, *args, **kwargs):# pylint: disable=unused-argument
        super().__init__()
        self.options = kwargs.get("options", {})
        self.layers = Sequential()
        self.skip_connection = False
        self.skip_layers = Identity()

    def forward(self, input_):# pylint: disable=arguments-differ
        if self.skip_connection:
            return self.skip_layers(input_) + self.layers(input_)
        return self.layers(input_)

    def get_number_params(self, trainable=True):
        """Retrieve the number of parameters for the module"""
        return sum(param.numel()
                   for param in self.parameters()
                   if param.requires_grad or not trainable)

    def __str__(self):
        string = super().__str__()
        string += f"\nTotal number of parameters: {self.get_number_params()}"
        return string
