"""Helpers for getting norms"""
from torch.nn import BatchNorm2d, InstanceNorm2d, LayerNorm
from .norms import AdaptiveInstanceNorm2d

AVAILABLE_NORMS = {
    "batch": BatchNorm2d,
    "instance": InstanceNorm2d,
    "layer": LayerNorm,
    "adain": AdaptiveInstanceNorm2d,
    }

def get_norm(norm_name):
    """Get a norm from a name"""
    return AVAILABLE_NORMS[norm_name]
