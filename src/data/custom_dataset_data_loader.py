# -*- coding: utf-8 -*-
"""Wrapper for creating datasets"""
import torch

from .aug_gan import create_aug_gan_dataset
from .cartoon_gan import create_cartoon_gan_dataset
from .source_target import SourceTargetDataset
from .multi_domain import MultiDomainDataset

AVAILABLE_DATASETS = {
    "SourceTarget": SourceTargetDataset,
    "CartoonGAN": create_cartoon_gan_dataset,
    "AugGAN": create_aug_gan_dataset,
    "MultiDomain": MultiDomainDataset,
    }

class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs
    multi-threaded data loading"""

    def __init__(self, options):
        """Initialize this class

        Create a dataset instance given the name [dataset_mode]
        Create a multi-threaded data loader.
        """
        self.options = options
        self.dataset = AVAILABLE_DATASETS[options.dataset_type](options)
        if options.verbose:
            print(f"Dataset [{type(self.dataset).__name__}] was created " +
                  f"with {len(self.dataset)}")
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=options.batch_size,
            #shuffle=not options.serial_batches,
            num_workers=int(options.num_threads))

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.options.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for index, data in enumerate(self.dataloader):
            if (index * self.options.batch_size >=
                    self.options.max_dataset_size):
                break
            yield data
