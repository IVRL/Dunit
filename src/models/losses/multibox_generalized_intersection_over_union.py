"""Generalized intersection over union for arbitrary lists of boxes"""
import torch
from torch.nn.functional import relu
from .generalized_intersection_over_union import \
    generalized_intersection_over_union
from .multibox_intersection_over_union import aux_multibox

def _get_term(gious, dimension, areas, nb_boxes, diff_nb_boxes,
              area_normalization):
    max_gious, _ = torch.max(gious, dimension)
    if area_normalization:
        max_gious = max_gious * areas / torch.sum(areas)
    ponderated_max_giou = relu(max_gious) / (nb_boxes + diff_nb_boxes) - \
        relu(-max_gious) * (1 + diff_nb_boxes) / nb_boxes
    return torch.sum(ponderated_max_giou)

def multibox_generalized_intersection_over_union(
        list1, list2, area_normalization=False):
    """Compute generalized intersection over union for two lists of boxes"""
    return aux_multibox(list1, list2, generalized_intersection_over_union,
                        _get_term, area_normalization)
