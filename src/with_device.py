"""Mixin for using the same device everywhere"""
import torch

class WithDeviceMixin():#pylint: disable=too-few-public-methods
    """Mixin for using the same device everywhere"""
    def __init__(self, *args, **kwargs):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(*args, **kwargs)
