"""Model for CycleGAN"""
from torch.nn import L1Loss

from ..base_model import BaseModel
from ...image_pool import ImagePool
from ..bidirectional_evaluation import BidirectionalEvaluationMixin
from ..losses import LSGANLoss
from .discriminator import CycleGANDiscriminator
from .generator import CycleGANGenerator

class CycleGAN(BidirectionalEvaluationMixin, BaseModel):
    """Main class for CycleGAN"""
    def __init__(self, options):
        super().__init__(options)
        # generators
        self.generators = [CycleGANGenerator(options) for _ in range(2)]

        # discriminators
        self.discriminators = [CycleGANDiscriminator(options)
                               for _ in range(2)]

        self.criterion_gan = LSGANLoss()
        self.criterion_cyc = L1Loss()

        self.source_pool = ImagePool(options.pool_size)
        self.target_pool = ImagePool(options.pool_size)

        self._end_setup(use_lists=True)

    def train_epoch(self, batch_source_images, batch_target_images):# pylint: disable=arguments-differ
        batch_source_images = batch_source_images[
            0]# batch contains (images, file_names)
        batch_target_images = batch_target_images[0]

        # get predicted labels for real images
        real_target_predicted_labels = self.discriminators[0](
            batch_target_images)
        real_source_predicted_labels = self.discriminators[1](
            batch_source_images)

        # generate transfered images
        fake_target_images = self.generators[0](batch_source_images)
        fake_source_images = self.generators[1](batch_target_images)
        # get predicted label from discriminator for generated images
        fake_target_predicted_labels = self.discriminators[0](
            self.target_pool.query(fake_target_images))
        fake_source_predicted_labels = self.discriminators[1](
            self.source_pool.query(fake_source_images))
        # get cycle-generated images
        cycle_source_images = self.generators[1](fake_target_images)
        cycle_target_images = self.generators[0](fake_source_images)

        # get discriminator loss
        discriminator_loss = self.criterion_gan.discriminator_loss(
            real_source_predicted_labels, fake_source_predicted_labels) + \
            self.criterion_gan.discriminator_loss(
                real_target_predicted_labels, fake_target_predicted_labels)
        # reduce learning of D with regards to G
        discriminator_loss = discriminator_loss * 0.5
        self._loss_backward(discriminator_loss, "discriminators", loss_id=0,
                            retain_graph=True)
        # get adversary loss
        adversary_loss = self.criterion_gan.generator_loss(
            fake_target_predicted_labels) + self.criterion_gan.generator_loss(
                fake_source_predicted_labels)
        # get cycle-consistency loss
        cycle_consistency_loss = self.criterion_cyc(
            batch_source_images, cycle_source_images) + self.criterion_cyc(
                batch_target_images, cycle_target_images)
        # get generator loss
        generator_loss = adversary_loss + \
            self.options.cycle_consistency_loss * cycle_consistency_loss
        self._loss_backward(generator_loss, "generators", loss_id=1)

        if self.options.verbose:
            self.discriminator_loss += discriminator_loss.item()
            self.generator_loss += generator_loss.item()

    def _evaluate_image(self, image, direction):
        transfer = self.generators[direction](image.unsqueeze(0))
        reconstruction = self.generators[self._negate_direction(direction)](
            transfer)
        return reconstruction, transfer

    @classmethod
    def update_arguments(cls, options):
        super().update_arguments(options)
        #### Loss weights ####
        options.parser.add_argument(
            '--cycle-consistency-loss-weight', type=float, default=10,
            dest="cycle_consistency_loss",
            help="Weight for cycle consistency loss")
