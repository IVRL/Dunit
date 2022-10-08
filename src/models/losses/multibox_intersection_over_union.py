"""Intersection over union for arbitrary lists of boxes"""
import torch
from .intersection_over_union import intersection_over_union, _get_area

def _get_term(ious, dimension, areas, nb_boxes, diff_nb_boxes,
              area_normalization):
    if ious.size(1) == 0 or ious.size(0) == 0:
        return 0
    max_ious, _ = torch.max(ious, dimension)
    if area_normalization:
        max_ious = max_ious * areas / torch.sum(areas)
    return torch.sum(max_ious) / (nb_boxes + diff_nb_boxes)

def aux_multibox(list1, list2, func, term_getter, area_normalization):
    """Generic function for turning a IoU loss in multibox version"""
    # compute areas for each box
    areas1 = torch.Tensor([_get_area(box) for box in list1])
    areas2 = torch.Tensor([_get_area(box) for box in list2])

    len1 = len(list1)
    len2 = len(list2)

    # compute IoU for each pair of boxes
    ious = torch.zeros(len1, len2)
    for index1, box1 in enumerate(list1):
        for index2, box2 in enumerate(list2):
            ious[index1, index2] = func(
                box1, box2, areas1[index1], areas2[index2])

    len_diff = abs(len1 - len2)
    return 0.5 * (
        term_getter(ious, 0, areas1, len1, len_diff, area_normalization) +
        term_getter(ious, 1, areas2, len2, len_diff, area_normalization))

def multibox_intersection_over_union(list1, list2, area_normalization=False):
    """Compute intersection over union for two lists of boxes"""
    return aux_multibox(list1, list2, intersection_over_union, _get_term,
                        area_normalization)
