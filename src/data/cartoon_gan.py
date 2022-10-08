# -*- coding: utf-8 -*-
"""Module creating a dataset as defined in the CartoonGAN paper"""
import os
import copy

import numpy as np
import cv2
from torchvision import transforms

from .source_target import SourceTargetDataset

def edge_smoothing(image_dataset, save, force=False):
    """Smooth edges in the image"""
    if not os.path.isdir(save):
        os.makedirs(save)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    for _, file_name in image_dataset:
        target_file_path = os.path.join(save, '{}.png'.format(
            ".".join(os.path.basename(file_name).split('.')[:-1])))
        if force or not os.path.exists(target_file_path):
            smooth_edges_image(file_name, kernel, kernel_size,
                               gauss, target_file_path)


def smooth_edges_image(file_name, kernel, kernel_size,
                       gauss, target_file_path):
    """Smooth edges for one image"""
    rgb_img = cv2.resize(cv2.imread(file_name), (256, 256))
    pad_img = np.pad(rgb_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')

    gray_img = cv2.imread(file_name, 0)
    gray_img = cv2.resize(gray_img, (256, 256))
    edges = cv2.Canny(gray_img, 100, 200)
    dilation = cv2.dilate(edges, kernel)

    gauss_img = np.copy(rgb_img)
    idx = np.where(dilation != 0)
    for i in range(np.sum(dilation != 0)):
        for j in range(3):
            gauss_img[idx[0][i], idx[1][i], j] = np.sum(
                np.multiply(
                    pad_img[idx[0][i]:idx[0][i] + kernel_size,
                            idx[1][i]:idx[1][i] + kernel_size, j],
                    gauss))

    cv2.imwrite(target_file_path, gauss_img)


def create_cartoon_gan_dataset(options):
    """Create a dataset according to the CartoonGAN paper"""
    if options.verbose:
        print("Creating CartoonGAN dataset")
    # first create a SourceTargetDataset
    dataset = SourceTargetDataset(
        options,
        source_transform=transforms.Compose([
            transforms.Resize((options.input_size, options.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]),
        target_transform=transforms.Compose([
            transforms.Resize((options.input_size, options.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        )
    # perform edge-smoothing on the target domain
    if options.verbose:
        print("Start edge smoothing")
    edge_smoothing(
        dataset.target,
        os.path.join(options.dataroot, options.target, 'edge-smooth'),
        options.force_image_preprocessing)
    if options.verbose:
        print("End edge smoothing")

    # use the pair edge-smoothed+normal as target dataset
    options_copy = copy.copy(options)
    options_copy.source = os.path.join(options.target, 'edge-smooth')
    target_dataset = SourceTargetDataset(
        options_copy,
        source_transform=transforms.Compose([
            transforms.Resize((options.input_size, options.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]),
        target_transform=transforms.Compose([
            transforms.Resize((options.input_size, options.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        )
    if options.verbose:
        print("Real dataset created")
    target_dataset.linked_datasets = True
    dataset.target = target_dataset
    return dataset
