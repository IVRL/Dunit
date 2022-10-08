"""Model for AugGAN"""
from torch.nn import ReflectionPad2d, InstanceNorm2d, L1Loss

from ..base_model import BaseModel
from ...image_pool import ImagePool
from ..bidirectional_evaluation import BidirectionalEvaluationMixin
from ..losses import GANLoss, SegmentationLoss
from .encoder import Encoder
from .multi_task_generator import MultiTaskGenerator
from .n_layer_discriminator import NLayerDiscriminator

class AugGAN(BidirectionalEvaluationMixin, BaseModel):
    """Model for AugGAN"""
    def __init__(self, options):
        super().__init__(options)
        self.encoders = [Encoder(
            self.options.nb_channels, self.options.last_layer_size,
            nb_blocks=self.options.nb_blocks, norm=InstanceNorm2d,
            padding_module=ReflectionPad2d).to(self.device) for _ in range(2)]

        self.decoders = [MultiTaskGenerator(
            4 * self.options.last_layer_size, self.options.nb_channels,
            parse_nc=32, norm=InstanceNorm2d, nb_blocks=6,
            padding_module=ReflectionPad2d).to(self.device) for _ in range(2)]

        self.discriminators = [NLayerDiscriminator(
            self.options.nb_channels, self.options.last_layer_size,
            n_layers=3, norm=InstanceNorm2d).to(self.device) for _ in range(2)]

        self.source_pool = ImagePool(options.pool_size)
        self.target_pool = ImagePool(options.pool_size)

        self.criterion_gan = GANLoss()
        self.criterion_cyc = L1Loss()
        self.criterion_seg = SegmentationLoss()
        self.generators = self.encoders + self.decoders

        self._end_setup(use_lists=True)

    def train_epoch(self, batch_source_images, batch_target_images):# pylint: disable=arguments-differ
        source_images, source_segmentation = batch_source_images[0]
        target_images, target_segmentation = batch_target_images[0]

        # encode images to features
        features_source = self.encoders[0](batch_source_images)
        features_target = self.encoders[1](batch_target_images)

        # generate images and segmentation from features
        fake_target_images, fake_source_segmentation = self.decoders[1](
            features_source)
        fake_source_images, fake_target_segmentation = self.decoders[0](
            features_target)

        # generate segmentation and reconstructed image from fakes
        source_reconstruction, _ = self.decoders[0](features_source)
        target_reconstruction, _ = self.decoders[1](features_target)

        # predict on real images
        source_prediction_real = self.discriminators[0](source_images)
        target_prediction_real = self.discriminators[1](target_images)
        # predict on generated images
        source_prediction_fake = self.discriminators[0](
            self.source_pool.query(fake_source_images))
        target_prediction_fake = self.discriminators[1](
            self.target_pool.query(fake_target_images))

        # compute GAN loss
        gan_loss = self.criterion_gan.generator_loss(source_prediction_fake) + \
            self.criterion_gan.generator_loss(target_prediction_fake)

        # compute segmentation loss
        segmentation_loss = self.criterion_seg(
            fake_source_segmentation, source_segmentation)[0] + \
            self.criterion_seg(fake_target_segmentation, target_segmentation)[0]

        # compute cycle-consistency loss
        cycle_consistency_loss = self.criterion_cyc(
            source_reconstruction, source_images) + \
            self.criterion_cyc(target_reconstruction, target_images)

        # compute soft weight sharing loss
        weight_sharing_loss = self.decoders[0].get_weight_sharing_loss() + \
            self.decoders[1].get_weight_sharing_loss()

        # compute total loss generator
        generator_loss = gan_loss + \
            self.options.segmentation_loss_weight * segmentation_loss + \
            self.options.cycle_consistency_loss * cycle_consistency_loss + \
            self.options.weight_sharing_loss_weight * weight_sharing_loss

        self._loss_backward(generator_loss, "generators", loss_id=0)

        # compute discriminator loss
        discriminator_loss_source = self.criterion_gan.discriminator_loss(
            source_prediction_real, source_prediction_fake)
        discriminator_loss_target = self.criterion_gan.discriminator_loss(
            target_prediction_real, target_prediction_fake)


        discriminator_loss = (
            discriminator_loss_source + discriminator_loss_target)
        self._loss_backward(discriminator_loss, "discriminators", loss_id=1)

        if self.options.verbose:
            self.discriminator_loss += discriminator_loss.item()
            self.generator_loss += generator_loss.item()

    def _evaluate_image(self, image, direction):
        features = self.encoders[direction](image.unsqueeze(0))
        reconstruction = self.decoders[direction](features)[0]
        transfer = self.decoders[not direction](features)[0]
        return reconstruction, transfer
