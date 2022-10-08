# -*- coding: utf-8 -*-
"""Dataset with multiple domains"""
from random import randint

from .base import BaseDataset
from .image_folder import ImageFolderDataset
from .utils import get_transform
from .coco_bbox import CocoBboxDataset
from .fake_annotation import FakeAnnotationDataset
from .init import INITDataset

AVAILABLE_ANNOTATED_DATASETS = {
    "coco_bbox": CocoBboxDataset,
    "": FakeAnnotationDataset,
    "None": FakeAnnotationDataset,
    "init": INITDataset,
    }

class MultiDomainDataset(BaseDataset):
    """Class representing a dataset with source and target domain"""
    def __init__(self, options):
        super().__init__(options)
        self.domains = {}
        if options.with_annotations:
            for (
                    domain_name, domain_folder,
                    domain_annotation_type, domain_annotation_folder) in zip(
                        options.domain_names, options.domain_folders,
                        options.domain_annotations,
                        options.domain_annotation_folders):
                transform = get_transform(options, None)
                self.domains[domain_name] = AVAILABLE_ANNOTATED_DATASETS[
                    domain_annotation_type](
                        options, folder=domain_folder, transform=transform,
                        annotation_folder=domain_annotation_folder)
        else:
            for domain_name, domain_folder in zip(
                    options.domain_names, options.domain_folders):
                transform = get_transform(options, None)
                self.domains[domain_name] = ImageFolderDataset(
                    options, folder=domain_folder, transform=transform)

        self.linked_datasets = self.options.serial_batches

    def __getitem__(self, index):
        """Return one image from each domain.

        Parameters
        ----------
        index: int
            Index of image

        Returns
        -------
        data: list of tuples
            [(domain_name, domain_image, domain_file_name)]
        """
        # make sure index is within the range
        for domain_index, domain_name in enumerate(self.domains.keys()):
            if domain_index == 0:
                data = [(
                    domain_name,
                    *self.domains[domain_name][
                        index % len(self.domains[domain_name])])]
            else:
                if self.linked_datasets:# use fixed pairs
                    index = index % len(self.domains[domain_name])
                else:
                    # randomize the index for other domains
                    # to avoid fixed pairs.
                    index = randint(0, len(self.domains[domain_name]) - 1)
                data.append((domain_name, *self.domains[domain_name][index]))
        return data

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of the size of the domains' datasets
        """
        return max(*[len(domain) for domain in self.domains.values()])
