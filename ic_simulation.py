#! /usr/bin/env python
"""Script to launch a linear regression learning bounding boxes"""
from collections import defaultdict
from datetime import timedelta
import json
import logging
from random import uniform, sample
import sys
from time import time

import numpy as np
import torch

from src.models.losses import IoULoss, AVAILABLE_IOU, multibox_iou

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

DEVICE = torch.device('cpu')
TYPE = torch.float32

def linear_regression(ground_truth, initial, nb_epochs, losses, height, width):
    """Linear regression for all losses"""
    predictions = {}
    errors = {}
    iou = {}
    for key_ in losses.keys():
        predictions[key_] = initial.clone().detach().requires_grad_(True)
        errors[key_] = []
        iou[key_] = []

    for epoch in range(nb_epochs):
        learning_rate = .1 if epoch < 160 else (.01 if epoch < 180 else 0.001)
        for key_ in losses.keys():
            if predictions[key_].grad is not None:
                predictions[key_].grad.zero_()
            loss = losses[key_](
                [{"boxes": ground_truth,
                  "labels": torch.tensor([#pylint: disable=not-callable
                      1 for _ in range(len(ground_truth))
                      ]).to(DEVICE)}],
                [{"boxes": predictions[key_],
                  "labels": torch.tensor([#pylint: disable=not-callable
                      1 for _ in range(len(predictions[key_]))
                      ]).to(DEVICE)}])
            errors[key_].append(loss.item())
            iou[key_].append(multibox_iou(
                ground_truth, predictions[key_]).item())
            loss.backward()
            with torch.no_grad():
                move = learning_rate * (height + width) * (
                    1 - multibox_iou(ground_truth, predictions[key_])
                    ) * predictions[key_].grad
                predictions[key_] -= move
    return errors, iou

def get_box(center, ratio, scale):
    """Create a box for a given center, ratio and scale"""
    width = 20 * scale * ratio
    height = 20 * scale / ratio
    return (center[0] - height/2, center[1] - width/2,
            center[0] + height/2, center[1] + width/2)

def build_boxes_list(centers, aspect_ratios, scales):
    """Build lists of boxes given centers, ratios and scale factors"""
    boxes_list = [[]]
    for center in centers:
        output_boxes_list = []
        for boxes in boxes_list:
            for ratio in aspect_ratios:
                for scale in scales:
                    list_ = boxes[:]
                    list_.append(get_box(center, ratio, scale))
                    output_boxes_list.append(list_)
        boxes_list = output_boxes_list
    return boxes_list

def _get_remaining_time(
        index, initial_boxes_list, gt_index, ground_truth_boxes_list,
        center_index, ground_truth_centers_list, epoch_time):
    #pylint: disable=redefined-outer-name
    for_current_gt = (1000//len(ground_truth_centers_list)) - index - 1#len(initial_boxes_list) - index - 1
    for_current_gt_center = (len(ground_truth_boxes_list) - gt_index - 1) * \
        (1000//len(ground_truth_centers_list))#len(initial_boxes_list)
    #for_remaining_gt_centers = len(ground_truth_boxes_list) * \
    #    1000 * \
    #    (len(ground_truth_centers_list) - center_index - 1)
    return timedelta(seconds=round(epoch_time * (
        for_current_gt + for_current_gt_center)))# + for_remaining_gt_centers)))

if __name__ == "__main__":
    EXAMPLE_INDEX = int(sys.argv[1])
    NAME = sys.argv[2]
    HEIGHT = 1000
    WIDTH = 1000
    NB_EPOCHS = 200
    USE_NAIVE_SOLUTION = EXAMPLE_INDEX > 0 and EXAMPLE_INDEX < 4
    LOSSES = {key: IoULoss(value)
              for key, value in AVAILABLE_IOU.items()
              if "simple" not in key and
              (USE_NAIVE_SOLUTION or "old_" not in key)}
    NB_POINTS = 100
    ASPECT_RATIOS = [np.sqrt(i) for i in [1, 4/3, 3/2, 2]]
    SCALES = [.5, 1., 2.]

    GROUND_TRUTH_CENTERS_LIST = [
        [(HEIGHT/2, WIDTH/2)],#only one box at the center
        # two boxes in two opposite corners
        [(HEIGHT/4, WIDTH/4), (3 * HEIGHT/4, 3 * WIDTH/4)],
        # two overlapping boxes in the center
        [(HEIGHT/2, WIDTH/2), (HEIGHT/2 + 30, WIDTH/2 + 30)],
        # 4 boxes in the center (overlapping areas for initial boxes)
        [(HEIGHT/2 - 30, WIDTH/2 - 30), (HEIGHT/2 - 30, WIDTH/2 + 30),
         (HEIGHT/2 + 30, WIDTH/2 - 30), (HEIGHT/2 + 30, WIDTH/2 + 30)],
        # 4 boxes (one in each corner)
        [(HEIGHT/4, WIDTH/4), (3 * HEIGHT/4, 3 * WIDTH/4),
         (HEIGHT/4, 3 * WIDTH/4), (3 * HEIGHT/4, WIDTH/4),],
        ]
    ERRORS_LIST = []
    IOUS_LIST = []
    try:
        #for center_index, ground_truth_centers in enumerate(
        #        GROUND_TRUTH_CENTERS_LIST):
            center_index = EXAMPLE_INDEX
            ground_truth_centers = GROUND_TRUTH_CENTERS_LIST[center_index]
            errors_storage = defaultdict(list)
            ious_storage = defaultdict(list)
            ground_truth_boxes_list = build_boxes_list(
                ground_truth_centers, ASPECT_RATIOS, [1])

            initial_boxes_list = []
            for _ in range(NB_POINTS):
                random_distance = 75 * uniform(0, 1)
                random_angle = 2 * np.pi * uniform(0, 1)
                initial_centers = [
                    (center[0] + random_distance * np.cos(random_angle),
                    center[1] + random_distance * np.sin(random_angle))
                    for center in ground_truth_centers]
                initial_boxes_list.extend(build_boxes_list(
                    initial_centers, ASPECT_RATIOS, SCALES))

            for gt_index, ground_truth_boxes in enumerate(ground_truth_boxes_list):
                ground_truth_boxes = torch.tensor(ground_truth_boxes,# pylint: disable=not-callable
                                                dtype=TYPE).to(DEVICE)
                for index, initial_boxes in enumerate(sample(initial_boxes_list, max(1000//len(ground_truth_boxes_list), 1))):
                    epoch_start_time = time()
                    # dict: loss_name -> list of errors for each epoch
                    error, iou_ = linear_regression(
                        ground_truth_boxes,
                        torch.tensor(initial_boxes, dtype=TYPE).to(DEVICE),# pylint: disable=not-callable
                        NB_EPOCHS, LOSSES, HEIGHT, WIDTH)
                    epoch_time = time() - epoch_start_time
                    string = ""
                    for key, value in error.items():
                        # dict: loss_name -> array of errors for each epoch and
                        #                    each trial
                        errors_storage[key].append(value)
                        ious_storage[key].append(iou_[key])
                        mean_for_loss = np.array(
                            ious_storage[key]).mean(0)[NB_EPOCHS - 1]
                        string += f", {key}:{mean_for_loss:.5f}"

                    expected_remaining_time = _get_remaining_time(
                        index, initial_boxes_list, gt_index,
                        ground_truth_boxes_list, center_index,
                        GROUND_TRUTH_CENTERS_LIST, epoch_time)
                    logging.info(
                        f"[{gt_index}/{len(ground_truth_boxes_list)}, " +
                        f"{index}/{(1000//len(ground_truth_centers_list))}] " +
                        f"Epoch: {epoch_time:.2f}, {expected_remaining_time} left" +
                        f"{string}")
            #ERRORS_LIST.append(errors_storage)
            #IOUS_LIST.append(ious_storage)
    except:
        pass
    finally:
        with open(f"results/regression/regression_errors_{NAME}.json", "w") as file_:
            json.dump(errors_storage, file_)

        with open(f"results/regression/regression_ious_{NAME}.json", "w") as file_:
            json.dump(ious_storage, file_)
