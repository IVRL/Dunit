"""Hinge loss for GANs"""
import torch

from .gan import GANLoss

class HingeGANLoss(GANLoss):
    """Hinge loss for GANs"""
    def _discriminator_loss(self, prediction_real, prediction_fake):
        return torch.min(
            torch.zeros_like(prediction_real), prediction_real - 0.5).mean() - \
            torch.max(torch.zeros_like(prediction_real),
                      prediction_fake + 0.5).mean()

    def generator_loss(self, prediction):
        return - prediction.mean()
