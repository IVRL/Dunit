"""Model for MUNIT"""
from collections import defaultdict
import torch
from torch.autograd import Variable
from torch.nn import InstanceNorm2d, L1Loss

from ..base_model import BaseModel
from ...image_pool import ImagePool
from ..multi_directional_evaluation import MultiDirectionalEvaluationMixin
from ..losses import LSGANLoss
from .ada_in_generator import AdaINGenerator
from ..unit import MultiScaleDiscriminator, UNITOptionsMixin

class MUNIT(UNITOptionsMixin, MultiDirectionalEvaluationMixin, BaseModel):
    """Main class for MUNIT model"""
    def __init__(self, options):
        self.nb_domains = len(options.domain_names)
        super().__init__(options)
        # Generators
        self.generators = {
            name: AdaINGenerator(options=options)
            for name in self.options.domain_names}

        # Discriminators
        self.discriminators = {
            name: MultiScaleDiscriminator(options=options)
            for name in self.options.domain_names}

        self.instance_norm = InstanceNorm2d(512, affine=False)
        self.style_dimension = options.style_dim

        self.pools = {
            name: ImagePool(options.pool_size)
            for name in self.options.domain_names}

        self.reconstruction_loss = L1Loss()
        self.gan_loss = LSGANLoss()

        self._end_setup()

    def train_epoch(self, *batch_domain_images):# pylint: disable=arguments-differ
        if len(batch_domain_images[0]) == 2:
            batch_domain_images = [
                (("source",), *batch_domain_images[0]),
                (("target",), *batch_domain_images[1])]

        discriminator_loss = 0
        generator_loss = 0

        data = defaultdict(lambda: defaultdict(dict))
        self._generate_training_data(batch_domain_images, data)

        generator_loss = self._generator_loss(data)
        discriminator_loss = self._discriminator_loss(data)

        self._loss_backward(
            discriminator_loss, "discriminators", loss_id=0, retain_graph=True)
        self._loss_backward(
            generator_loss, "generators", loss_id=1, retain_graph=True)
        if self.options.verbose:
            self.discriminator_loss += discriminator_loss.item()
            self.generator_loss += generator_loss.item()
        self._additional_training(data)
        return data

    def _generate_training_data(self, input_images, data):
        for domain_name, images, _ in input_images:
            domain_name = domain_name[0]
            data[domain_name]["real"] = images
            # encode (get features)
            data[domain_name]["features"], \
                style = self.generators[
                    domain_name].encode(images)
            # decode (within domain)
            data[domain_name]["reconstruction"] = self.generators[
                domain_name].decode(data[domain_name]["features"], style)
            data[domain_name]["random_style"] = Variable(
                torch.randn(images.size(0), self.style_dimension, 1, 1)
                .to(self.device))

        for domain_name in self.options.domain_names:
            # cross-domain data
            for target_name in self.options.domain_names:
                if target_name != domain_name:
                    cross_data = data[domain_name][target_name]
                    # decode (cross domain)
                    cross_data["fake"] = self.generators[target_name].decode(
                        data[domain_name]["features"],
                        data[domain_name]["random_style"])
                    # re-encode
                    content_reconstruct, cross_data["style_reconstruction"] = \
                        self.generators[target_name].encode(cross_data["fake"])
                    # re-decode
                    cross_data["cross_reconstruction"] = self.generators[
                        domain_name].decode(content_reconstruct, style)
                    cross_data["content_reconstruction"] = content_reconstruct

    def _discriminator_loss(self, data):
        gan_loss = 0
        for domain, domain_data in data.items():
            for target_domain in self.options.domain_names:
                if domain != target_domain:
                    gan_loss += self.discriminators[
                        target_domain].calc_dis_loss(
                            self.pools[target_domain].query(
                                domain_data[target_domain]["fake"].detach()),
                            data[target_domain]["real"])
        return gan_loss / (self.nb_domains * (self.nb_domains - 1))

    def _generator_loss(self, data):
        gan_loss = 0
        image_reconstruction_loss = 0
        style_reconstruction_loss = 0
        content_reconstruction_loss = 0
        cycle_consistency_loss = 0

        for domain, domain_data in data.items():
            # compute image reconstruction loss
            image_reconstruction_loss += self.reconstruction_loss(
                domain_data["reconstruction"], domain_data["real"])

            for target_domain in self.options.domain_names:
                if domain != target_domain:
                    # compute style reconstruction loss
                    style_reconstruction_loss += self.reconstruction_loss(
                        domain_data["random_style"],
                        domain_data[target_domain]["style_reconstruction"])
                    # compute content reconstruction loss
                    content_reconstruction_loss += self.reconstruction_loss(
                        domain_data["features"],
                        domain_data[target_domain]["content_reconstruction"])
                    # compute cycle consistency loss
                    cycle_consistency_loss += self.reconstruction_loss(
                        domain_data[target_domain]["cross_reconstruction"],
                        domain_data["real"])
                    # compute GAN loss
                    gan_loss += self.discriminators[
                        target_domain].calc_gen_loss(
                            domain_data[target_domain]["fake"])

        # compute global generator loss
        generator_loss = self.options.gan_loss_weight * gan_loss +\
            (self.options.content_reconstruction_loss_weight *
             content_reconstruction_loss) + \
            (self.options.style_reconstruction_loss_weight *
             style_reconstruction_loss) + \
            self.options.cycle_consistency_loss * \
            cycle_consistency_loss
        generator_loss /= self.nb_domains - 1
        generator_loss += (self.options.image_reconstruction_loss_weight *
                           image_reconstruction_loss)
        generator_loss /= self.nb_domains
        return generator_loss

    @classmethod
    def update_arguments(cls, options):
        super().update_arguments(options)
        #### Networks parameters ####
        options.parser.add_argument(
            '--mlp-dimension', type=int, dest="mlp_dim", default=256,
            help="""Imput dimension of the multi-layer perceptron use to
            compute the parameters of the AdaIN layer""")
        options.parser.add_argument(
            '--style-dimension', type=int, dest="style_dim", default=256,
            help="""Number of layers in the style encoder""")
        #### Loss weights ####
        options.parser.add_argument(
            '--image-reconstruction-loss-weight', type=float, default=10,
            dest="image_reconstruction_loss_weight",
            help="Weight for image reconstruction loss")
        options.parser.add_argument(
            '--content-reconstruction-loss-weight', type=float, default=0.01,
            dest="content_reconstruction_loss_weight",
            help="Weight for content reconstruction loss")
        options.parser.add_argument(
            '--style-reconstruction-loss-weight', type=float, default=0.01,
            dest="style_reconstruction_loss_weight",
            help="Weight for style reconstruction loss")

    def _evaluate_image(self, image, source_domain, target_domain):
        features, style = self.generators[source_domain].encode(
            image.unsqueeze(0))
        if source_domain != target_domain:
            style = Variable(
                torch.randn(1, self.style_dimension, 1, 1)
                .to(self.device))
        return self.generators[target_domain].decode(features, style)
