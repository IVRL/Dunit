# -*- coding: utf-8 -*-
"""Mixin for padding layers"""
from torch.nn import ZeroPad2d

class WithPaddingMixin():# pylint: disable=too-few-public-methods
    """Mixin for nets with a padding layer"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_layer = kwargs.get("padding_module", ZeroPad2d)(
            kwargs.get("padding", 1))
