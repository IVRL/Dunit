"""Utilities for DRIT"""
import torch

def merge_input_and_domain(input_, domains):
    """Prepare inputs for multi-modal nets"""
    domains = domains.view(domains.size(0), domains.size(1), 1, 1)
    domains = domains.repeat(1, 1, input_.size(2), input_.size(3))
    return torch.cat([input_, domains], dim=1)
