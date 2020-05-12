"""Model for object segmentation based on DRIT"""
from ..drit.model import DRIT
from .mixin import ObjectDetectionMixin

class DRITObjectDetection(
        ObjectDetectionMixin, DRIT):
    """Model for object segmentation based on DRIT"""
