# -*- coding: utf-8 -*-
"""Dataset corresponding to a folder of images"""
import os
import torch

from .image_folder import ImageFolderDataset

class ImageFolderWithAnnotationsDataset(ImageFolderDataset):#pylint: disable=too-few-public-methods
    """Class representing a dataset containing images with annotations"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{
            key: value for key, value in kwargs.items()
            if not key.startswith("annotation")})
        # create path to folder
        annotation_folder = kwargs.get(
            "annotation_folder", self.options.source_annotation)
        if annotation_folder is not None:
            self.annotation_path = os.path.join(
                self.options.dataroot,
                annotation_folder)

    def __getitem__(self, index):
        transformed_image, image_path = super().__getitem__(index)

        annotation = self._get_annotation(index, image_path)

        return (transformed_image, image_path, annotation)

    def _get_annotation(self, index, image_path):
        """Return the annotation for a given image"""
        raise NotImplementedError("_get_annotation not implemented")

    def to_local_tensor(self, data, dtype=torch.float):
        """Convert data to a Tensor on the model's device"""
        return torch.tensor(data, dtype=dtype).to(self.device)# pylint: disable=not-callable
