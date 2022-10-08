#! /usr/bin/env python
"""Script to launch a linear regression learning bounding boxes"""
from random import randint
import numpy as np
from matplotlib import pyplot as plt
import torch

from src.models.losses import multibox_generalized_iou, multibox_full_iou, \
    multibox_iou, IoULoss, multibox_complete_iou, \
    multibox_distance_iou

def get_random_bounding_box(height, width):
    """Get random coordinates for a bounding box"""
    left = randint(0, height - 1)
    bottom = randint(0, width - 1)
    right = randint(left, height - 1)
    top = randint(bottom, width - 1)
    return left, bottom, right, top

def _start(value):
    return max(int(value), 0)

def _end(value, max_value):
    return min(int(value), max_value)

def _limit(value, max_value):
    return _start(_end(value, max_value))

def print_box(image, box, color):
    """Display a box in an image"""
    width = image.shape[0]-1
    height = image.shape[1]-1
    image[_limit(box[0], width), _limit(box[1], height):
          _limit(box[3], height)] = color
    image[_limit(box[2], width), _limit(box[1], height):
          _limit(box[3], height)] = color
    image[_limit(box[0], width):
          _limit(box[2], width), _limit(box[1], height)] = color
    image[_limit(box[0], width):
          _limit(box[2] + 1, width), _limit(box[3], height)] = color

def show(height, width, box_lists):
    """Show image with all boxes"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for box_list in box_lists:
        for box in box_list[0]:
            print_box(image, box.detach(), box_list[1])
    plt.imshow(image)
    plt.title(",\n".join([
        box_list[2]
        for box_list in box_lists
        if box_list[2]
        ]))
    plt.show()

def linear_regression(height, width, nb_boxes1, nb_boxes2, nb_epochs, losses):
    """Main function for the linear regression"""
    ground_truth = torch.tensor(# pylint: disable=not-callable
        [get_random_bounding_box(height, width) for _ in range(nb_boxes1)],
        dtype=torch.float32)
    prediction = torch.tensor(# pylint: disable=not-callable
        [get_random_bounding_box(height, width) for _ in range(nb_boxes2)],
        dtype=torch.float32, requires_grad=True)
    predictions = {}
    for key in losses.keys():
        predictions[key] = prediction.clone().detach().requires_grad_(True)

    for epoch in range(nb_epochs):
        learning_rate = .1 if epoch < 160 else (.01 if epoch < 180 else 0.001)
        for key in losses.keys():
            if predictions[key].grad is not None:
                predictions[key].grad.zero_()
            loss = losses[key][0](
                [{"boxes": ground_truth,
                  "labels": torch.tensor([1 for _ in range(nb_boxes1)])}],# pylint: disable=not-callable
                [{"boxes": predictions[key],
                  "labels": torch.tensor([1 for _ in range(nb_boxes2)])}])# pylint: disable=not-callable
            loss.backward()
            with torch.no_grad():
                predictions[key] -= learning_rate * (height + width) * (
                    1 - multibox_iou(
                        ground_truth, predictions[key])
                    ) * predictions[key].grad

        print(f"[{epoch}/{nb_epochs}]")

    show(height, width, [
        (ground_truth, [0, 255, 0], f"Target: {ground_truth}"),
        (prediction, [255, 0, 0], f"Starting point: {prediction}")
        ] + [
            (predictions[key], losses[key][1],
             f"Box: {predictions[key]}, " +
             f"{key}: " +
             f"{multibox_iou(predictions[key], ground_truth)}, " +
             f"{multibox_generalized_iou(predictions[key], ground_truth)}")
            for key in losses.keys()
            ])

if __name__ == "__main__":
    HEIGHT = 100
    WIDTH = 100
    NB_EPOCHS = 200
    LOSSES = {
        #"MBIoU": (IoULoss(multibox_iou), [255, 255, 0]),#Yellow
        "MBGIoU": (IoULoss(multibox_generalized_iou), [0, 0, 255]),#blue
        #"MBL1": (IoULoss(multibox_l1), [0, 255, 255]),#Turquoise blue
        "MBDIoU": (IoULoss(multibox_distance_iou), [255, 0, 255]),#Magenta
        "MBCIoU": (IoULoss(multibox_complete_iou), [255, 255, 255]),#White
        "MBFIoU": (IoULoss(multibox_full_iou), [255, 255, 0]),#Yellow
        }
    linear_regression(HEIGHT, WIDTH, 1, 1, NB_EPOCHS, LOSSES)
    linear_regression(HEIGHT, WIDTH, 1, 2, NB_EPOCHS, LOSSES)
    linear_regression(HEIGHT, WIDTH, 2, 1, NB_EPOCHS, LOSSES)
    linear_regression(HEIGHT, WIDTH, 2, 2, NB_EPOCHS, LOSSES)
