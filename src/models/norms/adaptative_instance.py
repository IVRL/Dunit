# -*- coding: utf-8 -*-
"""Module implementing the Adaptative Instance Norm"""
import torch
from torch.nn import Module
from torch.nn.functional import batch_norm

from ..utils import attempt_use_apex

USE_APEX, _ = attempt_use_apex()

class AdaptiveInstanceNorm2d(Module):
    """Normalization layer according to the AdaIN definition"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input_):# pylint: disable=arguments-differ
        assert self.weight is not None and self.bias is not None, \
            "Please assign weight and bias before calling AdaIN!"
        size_x, size_y = input_.size(0), input_.size(1)
        running_mean = self.running_mean.repeat(size_x)
        running_var = self.running_var.repeat(size_x)

        # Apply instance norm
        input_reshaped = input_.contiguous().view(
            1, size_x * size_y, *input_.size()[2:])

        # if using mixed-precision optimization, need to cast tensors to float16
        if USE_APEX:
            running_mean = running_mean.half()
            running_var = running_var.half()
        out = batch_norm(
            input_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(size_x, size_y, *input_.size()[2:])

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.num_features})"
