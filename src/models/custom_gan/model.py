"""Model for AugGAN"""
from torch.nn import InstanceNorm2d, L1Loss

from ..base_model import BaseModel
from ...image_pool import ImagePool
from ..multi_directional_evaluation import MultiDirectionalEvaluationMixin
from ..losses import GANLoss, IoULoss
from ..aug_gan.n_layer_discriminator import NLayerDiscriminator
from .aug_gan_generator import AugGANGenerator

class CustomGAN(MultiDirectionalEvaluationMixin, BaseModel):
    """Model adapted from AugGAN for Object detection"""
    def __init__(self, options):
        super().__init__(options)
        if options.domain_names:
            self.domains = options.domain_names
        else:
            self.domains = ["source", "target"]

        # Generators
        self.generators = {
            name: AugGANGenerator(options=options)
            for name in self.domains}

        # Discriminators
        self.discriminators = {
            name: NLayerDiscriminator(
                self.options.nb_channels, self.options.last_layer_size,
                n_layers=3, norm=InstanceNorm2d).to(self.device)
            for name in self.domains}

        self.pools = {
            name: ImagePool(options.pool_size)
            for name in self.domains}

        self.criterion_gan = GANLoss()
        self.criterion_cyc = L1Loss()
        self.criterion_obj = IoULoss()

        self._end_setup()

    def train_epoch(self, *batch_domain_images):# pylint: disable=arguments-differ
        if not self.options.domain_names:
            batch_domain_images = [
                ("source", *batch_domain_images[0]),
                ("target", *batch_domain_images[1])]

        discriminator_loss = 0
        gan_loss = 0
        object_detection_loss = 0
        cycle_consistency_loss = 0

        for domain_name, images, annotations, __ in batch_domain_images:
            for target_name, target_images, _ in batch_domain_images:
                if target_name != domain_name:
                    # encode
                    features = self.generators[domain_name].encode(images)
                    # decode in same domain
                    reconstruction, predicted_annotations = \
                        self.generators[domain_name].decode(features)
                    # decode in target domain
                    fake_target, target_predicted_annotations = self.generators[
                        target_name].decode(features)
                    # predict on real images
                    prediction_real = self.discriminators[target_name](
                        target_images)
                    # predict on fake images
                    prediction_fake = self.discriminators[target_name](
                        self.pools[target_name].query(fake_target))
                    # compute GAN loss
                    gan_loss += self.criterion_gan.generator_loss(
                        prediction_fake)
                    # compute object detection loss
                    if annotations is not None:
                        object_detection_loss += self.criterion_obj(
                            annotations, predicted_annotations)
                    else:
                        object_detection_loss += self.criterion_obj(
                            target_predicted_annotations, predicted_annotations)
                    # compute cycle-consistency loss
                    cycle_consistency_loss += self.criterion_cyc(
                        reconstruction, images)
                    # compute discriminator loss
                    discriminator_loss += self.criterion_gan.discriminator_loss(
                        prediction_real, prediction_fake)

        # compute soft weight sharing loss
        weight_sharing_loss = sum([generator.decoder.get_weight_sharing_loss()
                                   for generator in self.generators.values()])

        # compute total loss generator
        generator_loss = self.options.gan_loss_weight * gan_loss + \
            self.options.object_detection_loss_weight * object_detection_loss +\
            self.options.cycle_consistency_loss * cycle_consistency_loss + \
            self.options.weight_sharing_loss_weight * weight_sharing_loss

        self._loss_backward(generator_loss, "generators", loss_id=0)
        self._loss_backward(discriminator_loss, "discriminators", loss_id=1)

        if self.options.verbose:
            self.discriminator_loss += discriminator_loss.item()
            self.generator_loss += generator_loss.item()

    def _evaluate_image(self, image, source_domain, target_domain):
        features = self.generators[source_domain].encode(image.unsqueeze(0))
        return self.generators[target_domain](features)[0]

    @classmethod
    def update_arguments(cls, options):
        """Add parameters for the model"""
        super().update_arguments(options)
        options.parser.add_argument(
            '--nb-downsample', type=int, dest="nb_downsample", default=2,
            help="Number of downsample layers for content encoder")
        options.parser.add_argument(
            '--nb-blocks', type=int, dest="nb_blocks", default=4,
            help="Number of ResBlocks in content encoder/decoder")
        options.parser.add_argument(
            '--nb-channels', type=int, dest="nb_channels", default=3,
            help="Number of channels in images")
        options.parser.add_argument(
            '--last-layer-size', type=int, dest="last_layer_size", default=64,
            help="Width of the bottom-most layer in the generator")
        #### Loss weights ####
        options.parser.add_argument(
            '--gan-loss-weight', type=float, dest="gan_loss_weight", default=1,
            help="Weight for GAN loss")
        options.parser.add_argument(
            '--object-detection-loss-weight', type=float, default=0.01,
            dest="object_detection_loss_weight",
            help="Weight for object detection loss")
        options.parser.add_argument(
            '--cycle-consistency-loss-weight', type=float, default=10,
            dest="cycle_consistency_loss",
            help="Weight for cycle consistency loss")
        options.parser.add_argument(
            '--weight-sharing-loss-weight', type=float, default=10,
            dest="weight_sharing_loss",
            help="Weight for weight sharing loss")
