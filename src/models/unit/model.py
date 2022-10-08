# -*- coding: utf-8 -*-
"""Model for UNIT"""
import torch
from torch.nn import InstanceNorm2d

from ..base_model import BaseModel
from ...image_pool import ImagePool
from ..utils import get_weight_init_func
from ..bidirectional_evaluation import BidirectionalEvaluationMixin
from .vae_generator import VAEGenerator
from .multi_scale_discriminator import MultiScaleDiscriminator
from .options import UNITOptionsMixin

def img_recon_criterion(img1, img2):
    """Compute reconstruction loss for images using MSE"""
    return torch.mean(torch.abs(img1 - img2))

def encoder_criterion(features):
    """Compute encoder loss"""
    return torch.mean(torch.pow(features, 2))

class UNIT(UNITOptionsMixin, BidirectionalEvaluationMixin, BaseModel):
    """Main class for UNIT model"""
    def __init__(self, options):
        super().__init__(options)
        # Generators
        self.generators = {
            "source": VAEGenerator(options=options),
            "target": VAEGenerator(options=options),
            }
        # Discriminators
        self.discriminators = {
            "source": MultiScaleDiscriminator(options=options),
            "target": MultiScaleDiscriminator(options=options),
            }

        self.instance_norm = InstanceNorm2d(512, affine=False)

        self.pools = {
            "source": ImagePool(options.pool_size),
            "target": ImagePool(options.pool_size)
            }

        self._end_setup()

    def _weight_initialization(self):
        for generator in self.generators.values():
            generator.apply(get_weight_init_func(self.options))
        for discriminator in self.discriminators.values():
            discriminator.apply(
                get_weight_init_func(self.options, type_='gaussian'))

    def train_epoch(self, batch_source_images, batch_target_images):# pylint: disable=arguments-differ
        images = {"source": {}, "target": {}}
        self._generate_training_images(batch_source_images,
                                       batch_target_images, images)

        # compute discriminator loss
        discriminator_loss = self.options.gan_loss_weight * \
            self._discriminator_loss(images)

        self._loss_backward(discriminator_loss, "discriminators", loss_id=0)

        # compute global generator loss
        generator_loss = self._generator_loss(images)

        self._loss_backward(generator_loss, "generators", loss_id=1)

        if self.options.verbose:
            self.discriminator_loss += discriminator_loss.item()
            self.generator_loss += generator_loss.item()

    def _generate_training_images(self, source_images, target_images, images):
        images["source"]["real"] = source_images[0]
        images["target"]["real"] = target_images[0]
        self._encode(images, "source")
        self._encode(images, "target")
        self._decode_same_domain(images, "source")
        self._decode_same_domain(images, "target")
        self._decode_cross_domain(images, "source")
        self._decode_cross_domain(images, "target")
        self._reencode(images, "source")
        self._reencode(images, "target")
        self._redecode(images, "source")
        self._redecode(images, "target")

    def _encode(self, images, domain):
        images[domain]["features"], images[domain]["noise"] = self.generators[
            domain].encode(images[domain]["real"])

    def _decode_same_domain(self, images, domain):
        images[domain]["reconstruction"] = self.generators[domain].decode(
            images[domain]["features"] + images[domain]["noise"])

    def _decode_cross_domain(self, images, domain):
        images[domain]["fake"] = self.generators[domain].decode(
            images[self._negate_direction(domain)]["features"] +
            images[domain]["noise"])

    def _reencode(self, images, domain):
        images[domain]["features_reconstruction"], \
            images[domain]["noise_reconstruction"] = \
                self.generators[self._negate_direction(domain)].encode(
                    images[self._negate_direction(domain)]["fake"])

    def _redecode(self, images, domain):
        images[domain]["cross_reconstruction"] = self.generators[domain].decode(
            images[domain]["features_reconstruction"] +
            images[domain]["noise_reconstruction"]) \
            if self.options.cross_reconstruction_loss_weight > 0 \
            else None

    def _discriminator_loss(self, data):
        return sum([self.discriminators[domain].calc_dis_loss(
            self.pools[domain].query(data[domain]["fake"].detach()),
            data[domain]["real"]) for domain in ["source", "target"]])

    def _image_reconstruction_loss(self, images, domain):#pylint: disable=no-self-use
        return img_recon_criterion(images[domain]["reconstruction"],
                                   images[domain]["real"])

    def _encoder_loss(self, images, domain):#pylint: disable=no-self-use
        return encoder_criterion(images[domain]["features"])

    def _cycle_encoder_loss(self, images, domain):#pylint: disable=no-self-use
        return encoder_criterion(images[domain]["features_reconstruction"])

    def _cycle_consistency_loss(self, images, domain):#pylint: disable=no-self-use
        return img_recon_criterion(images[domain]["cross_reconstruction"],
                                   images[domain]["real"])

    def _gan_loss(self, images, domain):
        return self.discriminators[domain].calc_gen_loss(
            images[domain]["fake"])

    def _generator_loss(self, data):
        gan_loss = self._gan_loss(data, "source") + self._gan_loss(
            data, "target")
        image_reconstruction_loss = self._image_reconstruction_loss(
            data, "source") + self._image_reconstruction_loss(
                data, "target")
        encoder_loss = self._encoder_loss(data, "source") + \
            self._encoder_loss(data, "target")
        cycle_encoder_loss = self._cycle_encoder_loss(data, "source") + \
            self._cycle_encoder_loss(data, "target")
        cycle_consistency_loss = self._cycle_consistency_loss(data, "source")\
            + self._cycle_consistency_loss(data, "target")
        return self.options.gan_loss_weight * gan_loss + \
            (self.options.reconstruction_loss_weight *
             image_reconstruction_loss) + \
            self.options.encoder_loss_weight * encoder_loss + \
            self.options.cycle_encoder_loss_weight * cycle_encoder_loss + \
            self.options.cycle_consistency_loss * cycle_consistency_loss

    @classmethod
    def update_arguments(cls, options):
        super().update_arguments(options)
        #### Loss weights ####
        options.parser.add_argument(
            '--cross-reconstruction-loss-weight', type=float, default=10,
            dest="cross_reconstruction_loss_weight",
            help="Weight for cross reconstruction loss")
        options.parser.add_argument(
            '--encoder-loss-weight', type=float, default=0.01,
            dest="encoder_loss_weight",
            help="Weight for encoder loss")
        options.parser.add_argument(
            '--reconstruction-loss-weight', type=float, default=10,
            dest="reconstruction_loss_weight",
            help="Weight for reconstruction loss")

    def _evaluate_image(self, image, direction):
        features, _ = self.generators[direction].encode(image.unsqueeze(0))
        reconstruction = self.generators[direction].decode(features)
        transfer = self.generators[self._negate_direction(direction)].decode(
            features)
        return reconstruction, transfer
