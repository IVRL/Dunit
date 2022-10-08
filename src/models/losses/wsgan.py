"""Loss for WSGAN"""
from .gan import GANLoss

class WSGANLoss(GANLoss):
    """Loss for WSGAN"""
    def _discriminator_loss(self, prediction_real, prediction_fake):
        return prediction_fake.mean() - prediction_real.mean()

    def generator_loss(self, prediction):
        return - prediction.mean()
