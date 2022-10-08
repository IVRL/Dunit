"""Loss for traditional GAN"""
import torch
from torch.autograd import grad
from torch.nn import BCEWithLogitsLoss

PENALTY_EXPRESSIONS = {
    "gradient": lambda norm: ((norm - 1) ** 2).mean(),
    "lipschitz": lambda norm: (
        torch.max(torch.zeros_like(norm), norm - 1) ** 2).mean(),
    }

class GANLoss():
    """Vanilla loss for GANs"""
    def __init__(self, clipping_value=0, gradient_penalty_weight=0,
                 penalty_type="gradient", sample_mode=None, norm=2):
        self.loss = BCEWithLogitsLoss()
        self.clipping_value = clipping_value
        self.gradient_penalty_weight = gradient_penalty_weight
        self.penalty_type = penalty_type
        self.sample_mode = sample_mode
        self.norm = norm

    def discriminator_loss(self, prediction_real, prediction_fake,
                           discriminator=None, real_data=None, fake_data=None):
        """Return full loss for the discriminator"""
        prediction_loss = self._discriminator_loss(
            prediction_real, prediction_fake)
        if self.gradient_penalty_weight == 0:
            return  prediction_loss
        return prediction_real + self._gradient_penalty(
            discriminator, real_data, fake_data) * self.gradient_penalty_weight

    def _discriminator_loss(self, prediction_real, prediction_fake):
        """Return loss for the discriminator on prediction only"""
        return self.loss(prediction_real, torch.ones_like(prediction_real)) + \
            self.loss(prediction_fake, torch.zeros_like(prediction_fake))

    def generator_loss(self, prediction):
        """Return loss for the generator"""
        return self.loss(prediction, torch.ones_like(prediction))

    def gradient_clipping(self, discriminator):
        """Gradient clipping"""
        if self.clipping_value > 0:
            for parameter in discriminator.parameters():
                parameter.data.clamp_(
                    -self.clipping_value, self.clipping_value)

    def _gradient_penalty(self, discriminator, real_data, fake_data):
        other_data = (
            real_data + 0.5 * real_data.std() * torch.rand_like(real_data)
            if self.sample_mode == 'dragan'
            else fake_data)
        shape = [real_data.size(0)] + [1] * (real_data.dim() - 1)
        alpha = torch.rand(shape, device=real_data.device)
        sampled_data = real_data - alpha * (other_data - real_data)
        sampled_data.requires_grad = True
        prediction = discriminator(sampled_data)
        gradient = grad(prediction, sampled_data,
                        grad_outputs=torch.ones_like(prediction),
                        create_graph=True, retain_graph=True,
                        only_inputs=True)[0]
        gradient_norm = gradient.view(gradient.size(0), -1).norm(
            p=self.norm, dim=1)

        return PENALTY_EXPRESSIONS[self.penalty_type](gradient_norm)
