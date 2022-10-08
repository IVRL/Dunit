"""Generalized intersection over union"""
import torch
from .intersection_over_union import intersection_over_union, _get_area

def _get_square_envelop(box1, box2):
    """Retrieve the closest square containing two boxes"""
    return (torch.min(box1[0], box2[0]), torch.min(box1[1], box2[1]),
            torch.max(box1[2], box2[2]), torch.max(box1[3], box2[3]))

def generalized_intersection_over_union(box1, box2, area1=None, area2=None):
    """Compute the generalized intersection over union score
    for the input boxes"""
    if area1 is None:
        area1 = _get_area(box1)
    if area2 is None:
        area2 = _get_area(box2)
    iou = intersection_over_union(box1, box2, area1, area2)
    convex_envelop = _get_square_envelop(box1, box2)
    convex_envelop_area = _get_area(convex_envelop)

    return iou - ((convex_envelop_area - area1 - area2) / convex_envelop_area)
