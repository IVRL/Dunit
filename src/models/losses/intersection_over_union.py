"""Intersection over union"""
import torch

def _get_area(box):
    """Compute the area of a box"""
    return (box[2] - box[0]) * (box[3] - box[1])

def _get_intersection(box1, box2):
    """Retrieve the intersection of two boxes"""
    return (torch.max(box1[0], box2[0]), torch.max(box1[1], box2[1]),
            torch.min(box1[2], box2[2]), torch.min(box1[3], box2[3]))

def intersection_over_union(box1, box2, area1=None, area2=None):
    """Return the intersection over union score for the input boxes

    Arguments:
    ----------
    box1, box2: tuple/list (x1, y1, x2, y2)
        Boxes to compare

    Returns:
    --------
    iou: float
        Intersection over union score
    """
    intersection = _get_intersection(box1, box2)
    if intersection[0] > intersection[2] or intersection[1] > intersection[3]:
        return 0

    if area1 is None:
        area1 = _get_area(box1)
    if area2 is None:
        area2 = _get_area(box2)
    area_intersection = _get_area(intersection)

    return area_intersection / (area1 + area2 - area_intersection)
