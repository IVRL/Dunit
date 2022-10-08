# -*- coding: utf-8 -*-
"""Various functions for data loading"""
import os
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomCrop, \
    RandomHorizontalFlip, RandomVerticalFlip, Resize, ToTensor
from PIL import Image

def make_dataset(directory, max_dataset_size=float("inf"), recursive=False):
    """Get list of images from folder"""
    images = []
    assert os.path.isdir(directory), f"{directory} is not a valid directory"

    index = 0
    for directory_path, _, file_names in os.walk(directory, followlinks=True):
        for file_name in file_names:
            if index < max_dataset_size and \
                    os.path.splitext(file_name)[-1] in IMG_EXTENSIONS:
                index += 1
                path = os.path.join(directory_path, file_name)
                images.append(path)
            if index >= max_dataset_size:
                return sorted(images)
        if not recursive:
            break
    return sorted(images)

def get_transform(options, transform=None):
    """Return tranformation from either options or parameter"""
    if transform is not None:
        return transform

    transform_list = []

    if options.center_crop:
        transform_list.append(
            CenterCrop((options.crop_height, options.crop_width)))
    elif options.crop:
        transform_list.append(
            RandomCrop((options.crop_height, options.crop_width)))
    if options.input_size:
        transform_list.append(Resize((options.input_size, options.input_size)))
    if options.flip_horizontal:
        transform_list.append(RandomHorizontalFlip(options.proba_hori_flip))
    if options.flip_vertical:
        transform_list.append(RandomVerticalFlip(options.proba_vert_flip))
    transform_list.append(ToTensor())
    if options.normalize:
        transform_list.append(
            Normalize(options.normalize_mean, options.normalize_std))
    return Compose(transform_list)

def get_segmentation_annotation(options, path, transform):
    """Return annotation getter for segmentation"""
    file_paths = make_dataset(path, options.max_dataset_size)

    transform = get_transform(options, transform)

    for transform_func in transform:
        if "Resize" in transform_func.__class__.__name__:
            transform_func.interpolation = Image.NEAREST
        if "Random" in transform_func.__class__.__name__:
            raise NotImplementedError(
                "Cannot use random transformations with annotations""")

    def getter(index, *args):#pylint: disable=unused-argument
        """Getter for one annotation"""
        image = Image.open(file_paths[index])
        return transform(image)

    return getter

AVAILABLE_ANNOTATIONS = {
    "segmentation": get_segmentation_annotation,
    }

def get_annotation_getter(options, annotation_type, path, transform=None):
    """Retrieve the getter for annotations"""
    return AVAILABLE_ANNOTATIONS[
        annotation_type
        if annotation_type is not None
        else options.annotation_type](options, path, transform)
