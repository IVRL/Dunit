#! /usr/bin/env python
"""Illustrate the impact of various IoU related losses on boxes
shapes and positions
"""
from time import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from src.models.losses import AVAILABLE_IOU

HEIGHT = 100
WIDTH = 100
GROUND_TRUTH = torch.tensor([(10, 10, 20, 20), (80, 80, 90, 90)],# pylint: disable=not-callable
                            dtype=torch.float32)

INDEXES = np.arange(-.1, 1.1, 0.005)
BOXES_LIST = [
    torch.tensor([(10, 10, 20, 20), (# pylint: disable=not-callable
        index * 10 + (1 - index) * 80,
        index * 10 + (1 - index) * 80,
        (np.sqrt(np.sqrt(abs(index)))) * 20 + (1 -  index) * 90,
        (index ** 4) * 20 + (1 -  index) * 90,
        )])
    for index in INDEXES]
START_TIME = time()
VALUES = {}
DURATIONS = {}
for key, value in AVAILABLE_IOU.items():
    START_TIME_IOU = time()
    VALUES[key] = [1 - value(GROUND_TRUTH, boxes)
                   for boxes in BOXES_LIST]
    DURATIONS[key] = time() - START_TIME_IOU
DURATION = time() - START_TIME
print(DURATION, DURATION / (len(AVAILABLE_IOU) * len(BOXES_LIST)))
for key, value in DURATIONS.items():
    if f"old_{key}" in DURATIONS.keys():
        overhead = (value - DURATIONS[f'old_{key}']) / DURATIONS[f'old_{key}']
        print(f"{key} -> Old: {DURATIONS['old_' + key]}, new: {value}, "
              f"overhead: {overhead:.2%}")

TO_PLOT = [
    (INDEXES, values, color) for values, color in zip(
        VALUES.values(), ['c', 'r', 'b', 'k', 'y', 'g', 'm', 'orange',
                          'lime', 'navy', 'chocolate', 'darkviolet'])
    ]

plt.plot(
    *[i for l in TO_PLOT for i in l]
    )
plt.show()
