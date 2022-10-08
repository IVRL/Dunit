"""Loss for segmentation"""
import torch
from torch.nn import CrossEntropyLoss, L1Loss, LogSoftmax

class SegmentationLoss():
    """Loss for segmentation"""
    def __init__(self, l1_weight=0.0, weights=None):
        self.l1_criterion = L1Loss()
        self.cross_entropy = CrossEntropyLoss(weight=weights)
        self.l1_weight = l1_weight
        self.log_softmax = LogSoftmax()

    def __call__(self, pred, target):
        return self.pixelwise_loss(pred, target)

    def pixelwise_loss(self, prediction, target):
        """Pixel-wise loss"""
        cross_entropy_loss = self.cross_entropy(prediction, target)

        if not self.l1_weight:
            return cross_entropy_loss

        onehot_target = torch.zeros_like(prediction).scatter_(
            1, target.data.unsqueeze(1), 1)

        l1_loss = self.l1_criterion(prediction, onehot_target)

        return cross_entropy_loss + self.l1_weight * l1_loss
