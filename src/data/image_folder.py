# -*- coding: utf-8 -*-
"""Dataset corresponding to a folder of images"""
import os
from PIL import Image, ImageFile

from .base import BaseDataset
from .utils import make_dataset, get_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageFolderDataset(BaseDataset):
    """Class representing a dataset with image and target domain"""
    def __init__(self, options, folder=None, transform=None):
        super().__init__(options)
        # create path to folder
        self.folder_path = os.path.join(
            options.dataroot, folder if folder is not None else options.source)
        # load images from 'folder'
        self.image_paths = make_dataset(
            self.folder_path, options.max_dataset_size, options.recursive)
        # get the size of the dataset
        self.size = len(self.image_paths)
        self.transform = get_transform(options, transform)

    def __getitem__(self, index):
        """Return one image from folder.

        Parameters
        ----------
        index: int
            Index of image

        Returns
        -------
        data: tuple
            (image, file_name)
        """
        # make sure index is within the range
        image_path = self.image_paths[index % self.size]
        image = Image.open(image_path).convert('RGB')
        # apply image transformation
        transformed_image = self.transform(image).to(self.device)

        return (transformed_image, image_path)

    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return self.size
