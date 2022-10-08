"""Loss for Intersection over Union"""
import torch
from torch.nn import _reduction
from torch.nn.modules.loss import _Loss

from .multibox_intersection_over_union import multibox_intersection_over_union

def _filter_boxes_for_class(input_, class_):
    """Keep only boxes from the given class"""
    return [box for box, label in zip(input_['boxes'],
                                      input_['labels'])
            if label == class_]

def iou_loss(input_, target, iou_func=multibox_intersection_over_union,
             size_average=None, reduce_=None, reduction='mean',
             area_normalization=False):
    """Loss function for multi-class IoU

    Arguments
    ---------
    input_, target: list of dicts
        Prediction and reference value for a list of images
        For each image, we expect a dictionary containing:
        - boxes: the list of all detected boxes
        - labels: the list of labels for the boxes
        - scores: the list of scores for the boxes
    area_normalization: bool
        True to normalize the weight of a box in the multibox IoU metric by its
        relative area in the image compared to all the boxes
    """
    if size_average is not None or reduce_ is not None:
        reduction = _reduction.legacy_get_string(size_average, reduce_)
    per_image_iou = []
    for prediction, ground_truth in zip(input_, target):# for each image
        # get the list of found classes
        classes = torch.cat((prediction['labels'],
                             ground_truth['labels'])).unique()
        per_image_iou.append(torch.Tensor([
            1 - iou_func(
                _filter_boxes_for_class(prediction, class_),
                _filter_boxes_for_class(ground_truth, class_),
                area_normalization
                ) for class_ in classes]).mean())
    ret = torch.Tensor(per_image_iou)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

class IoULoss(_Loss):
    """Loss module for IoU"""
    __constants__ = ['reduction']

    def __init__(self, iou_func=multibox_intersection_over_union,
                 size_average=None, reduce_=None, reduction='mean',
                 area_normalization=False):
        super().__init__(size_average, reduce_, reduction)
        self.iou_func = iou_func
        self.area_normalization = area_normalization

    def forward(self, input_, target):# pylint: disable=arguments-differ
        return iou_loss(input_, target, reduction=self.reduction,
                        iou_func=self.iou_func,
                        area_normalization=self.area_normalization)
