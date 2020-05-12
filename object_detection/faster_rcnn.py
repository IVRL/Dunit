"""Faster RCNN with object detection loss"""
from ..faster_rcnn import FasterRCNNModel
from .mixin import ObjectDetectionModelMixin

class FasterRCNNObjectDetection(ObjectDetectionModelMixin, FasterRCNNModel):
    """Faster RCNN with object detection loss"""
