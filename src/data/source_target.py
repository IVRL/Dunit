# -*- coding: utf-8 -*-
"""Dataset with source and target domains"""
from random import randint

from .base import BaseDataset
from .image_folder import ImageFolderDataset
from .utils import get_transform
from .multi_domain import AVAILABLE_ANNOTATED_DATASETS

class SourceTargetDataset(BaseDataset):
    """Class representing a dataset with source and target domain"""
    def __init__(self, options, source_transform=None,
                 target_transform=None, with_annotations=False):
        super().__init__(options)
        source_transform = get_transform(options, source_transform)
        target_transform = get_transform(options, target_transform)
        if with_annotations or options.with_annotations:
            self.source = AVAILABLE_ANNOTATED_DATASETS[
                options.source_annotation_type](
                    options, folder=options.source, transform=source_transform,
                    annotation_folder=options.source_annotation)
            self.target = AVAILABLE_ANNOTATED_DATASETS[
                options.target_annotation_type](
                    options, folder=options.source, transform=source_transform,
                    annotation_folder=options.source_annotation)
        else:
            self.source = ImageFolderDataset(
                options, folder=options.source, transform=source_transform)
            self.target = ImageFolderDataset(
                options, folder=options.target, transform=target_transform)

        self.linked_datasets = self.options.serial_batches

    def __getitem__(self, index):
        """Return one image from both source + name.

        Parameters
        ----------
        index: int
                Index of image

        Returns
        -------
        data: tuple of tuples
            (source_image, source_file_name), (target_image, target_image_name)
        """
        # make sure index is within the range
        source_data = self.source[index % len(self.source)]
        if self.linked_datasets:# use fixed pairs
            index_target = index % len(self.target)
        else:# randomize the index for target domain to avoid fixed pairs.
            index_target = randint(0, len(self.target) - 1)
        target_data = self.target[index_target]
        return source_data, target_data

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of the size of the domains' datasets
        """
        return max(len(self.source), len(self.target))
