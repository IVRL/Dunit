"""Model for object segmentation based on UNIT"""
from ..unit.model import UNIT
from .mixin import ObjectDetectionMixin

class UNITObjectDetection(
        ObjectDetectionMixin, UNIT):
    """Model for object segmentation based on UNIT"""
