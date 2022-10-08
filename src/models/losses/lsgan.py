"""Loss for LSGAN"""
from torch.nn import MSELoss

from .gan import GANLoss

class LSGANLoss(GANLoss):
    """Loss for LSGAN"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = MSELoss()
