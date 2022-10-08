"""Mixin for using a content discriminator"""
import torch
from ..utils import attempt_use_apex

USE_APEX, AMP = attempt_use_apex()

class ContentDiscriminatorMixin():
    """Mixin for using a content discriminator"""
    def __init__(self, options, content_discriminator):
        self.content_discriminator_loss = 0
        self.content_discriminator = content_discriminator(
            options=options, nb_domains=self.nb_domains)
        super().__init__(options)
        self.content_discriminator.to(self.device)

    def create_optimizers(self, use_lists):
        """Override creating optimizers"""
        super().create_optimizers(use_lists)
        self._create_optimizer([self.content_discriminator], True,
                               "content_discriminator")


    def _set_nets(self, use_lists):
        super()._set_nets(use_lists)
        self.nets.append(self.content_discriminator)

    def _initialize_apex(self, use_lists):
        super()._initialize_apex(use_lists)
        if USE_APEX and self.options.use_mixed_precision:
            self.nets, [
                self.optimizers["content_discriminator"],
                ] = AMP.initialize(
                    self.nets,
                    [self.optimizers["content_discriminator"]],
                    opt_level="O1", num_losses=1)
            i = 0
            for dictionary in [self.generators, self.discriminators]:
                for key in (range(len(dictionary)) if use_lists
                            else dictionary.keys()):
                    dictionary[key] = self.nets[i]
                    i += 1

    def _additional_training(self, data):
        super()._additional_training(data)
        content_discriminator_loss = self._content_discriminator_loss(data)
        self._loss_backward(
            content_discriminator_loss, "content_discriminator", loss_id=2)
        if self.options.verbose:
            self.content_discriminator_loss += content_discriminator_loss.item()

    def _generate_training_data(self, input_images, data):
        super()._generate_training_data(input_images, data)

        for domain_name in self.options.domain_names:
            data[domain_name]["content_prediction"] = \
                self.content_discriminator(
                    data[domain_name]["content"].detach())

    def _generator_loss(self, data):
        generator_loss = super()._generator_loss(data)
        content_adversary_loss = 0
        for domain_data in data.values():
            # content adversary loss
            # Warning: official DRIT++ implementation uses (1 - domain_encoding)
            # as target, contrary to what is presented in the paper
            # DRIT official implementation is correct
            content_adversary_loss += self.classification_loss(
                domain_data["content_prediction"],
                0.5 * torch.ones_like(domain_data["content_prediction"]))
        generator_loss += (
            content_adversary_loss *
            self.options.content_adversary_loss_weight) / self.nb_domains
        return generator_loss

    def _content_discriminator_loss(self, data):
        content_discriminator_loss = 0
        for domain_data in data.values():
            content_discriminator_loss += self.classification_loss(
                domain_data["content_prediction"], domain_data["encoding"])
        return content_discriminator_loss / self.nb_domains

    def log_end_epoch(self, nb_iterations):
        """String to display at the end of an epoch"""
        string = super().log_end_epoch(nb_iterations)
        content_discriminator_loss = self.content_discriminator_loss / \
            nb_iterations
        string += (
            f", Content discriminator loss: {content_discriminator_loss:.3f}")
        self.content_discriminator_loss = 0
        return string

    @classmethod
    def update_arguments(cls, options):
        """Update arguments"""
        super().update_arguments(options)
        options.parser.add_argument(
            '--learning-rate-content-discriminator', type=float, default=0.0002,
            dest='learning_rate_content_discriminator',
            help="Learning rate for the content discriminator")
        options.parser.add_argument(
            '--content-adversary-loss-weight', type=float, default=1,
            dest="content_adversary_loss_weight",
            help="Weight for content adversary loss")
