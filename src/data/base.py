# -*- coding: utf-8 -*-
"""Base dataset"""
from torch.utils.data import Dataset

from ..with_device import WithDeviceMixin

class BaseDataset(WithDeviceMixin, Dataset):
    """Base dataset"""
    def __init__(self, options):
        super().__init__()
        self.options = options
        self.size = 0

    def __getitem__(self, index):
        raise NotImplementedError("__getitem__ not implemented for dataset")

    def __iter__(self):
        """Return a batch of data"""
        for index in range(self.size):
            yield self[index]
