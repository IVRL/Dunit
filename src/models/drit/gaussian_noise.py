"""Layer returning Gaussian noise"""
import torch
from torch.autograd import Variable

from ..base_module import BaseModule

class GaussianNoiseLayer(BaseModule):
    """Layer returning Gaussian noise"""
    def forward(self, input_):
        if self.training:
            return input_ + Variable(torch.randn(input_.shape).to(self.device))
        return input_
