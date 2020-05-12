"""Main model for DRIT"""
from collections import defaultdict
import torch
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.autograd import Variable

from ..base_model import BaseModel
from ..multi_directional_evaluation import MultiDirectionalEvaluationMixin
from .concat_generator import ConcatGenerator
from .generator import Generator
from .multi_scale_discriminator import MultiScaleDiscriminator
from .discriminator import Discriminator
from .content_discriminator import ContentDiscriminator
from ..losses import GANLoss
from ..mixins import ContentDiscriminatorMixin
from ..utils import attempt_use_apex

USE_APEX, AMP = attempt_use_apex()

class DRIT(ContentDiscriminatorMixin, MultiDirectionalEvaluationMixin,
           BaseModel):
    """Main model for DRIT"""
    def __init__(self, options):
        self.nb_domains = len(options.domain_names)
        super().__init__(options, content_discriminator=ContentDiscriminator)

        # Generators
        self.generator = (ConcatGenerator if options.concat else Generator)(
            options=options, nb_domains=self.nb_domains)

        # Discriminators
        self.discriminator = (MultiScaleDiscriminator if options.nb_scales > 1
                              else Discriminator)(options=options,
                                                  nb_domains=self.nb_domains)

        self.classification_loss = BCEWithLogitsLoss()
        self.gan_loss = GANLoss()
        self.l1_loss = L1Loss()

        self._end_setup()

    def create_optimizers(self, use_lists):
        for name, nets in [("generator", [self.generator]),
                           ("discriminator", [self.discriminator])]:
            plural_name = f"{name}s"
            self._create_optimizer(nets, True, plural_name, name)
        self._create_optimizer([self.content_discriminator], True,
                               "content_discriminator")

    def _set_nets(self, use_lists):
        self.nets = [self.generator, self.discriminator,
                     self.content_discriminator]

    def _initialize_apex(self, use_lists):
        # use apex optimizations
        if USE_APEX and self.options.use_mixed_precision:
            [self.generator, self.discriminator, self.content_discriminator], [
                self.optimizers["generators"],
                self.optimizers["discriminators"],
                self.optimizers["content_discriminator"],
                ] = AMP.initialize(
                    [self.generator, self.discriminator,
                     self.content_discriminator],
                    [self.optimizers["generators"],
                     self.optimizers["discriminators"],
                     self.optimizers["content_discriminator"],],
                    opt_level="O1", num_losses=3)

    def train_epoch(self, *batch_domain_images):#pylint: disable=arguments-differ
        if not isinstance(batch_domain_images[0][0][0], str):
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
            images = images["image"]
            data[domain_name]["real"] = images
            encoding = torch.zeros((images.size(0), self.nb_domains)).to(
                self.device)
            encoding[:, self.options.domain_names.index(domain_name)] = 1
            data[domain_name]["encoding"] = encoding
            content, style = self.generator.encode(images, encoding)
            data[domain_name]["content"] = content
            style, mean, logvar = self._get_style(style)
            data[domain_name]["style"] = style
            if self.options.concat:
                data[domain_name]["mean"] = mean
                data[domain_name]["logvar"] = logvar
            # decode to own domain
            data[domain_name]["reconstruction"] = self.generator.decode(
                content, style, encoding)
            data[domain_name]["style_reconstruction"], _, _ = self._get_style(
                self.generator.encode_style(
                    data[domain_name]["reconstruction"], encoding))
            # predictions
            data[domain_name]["real_gan_prediction"], data[domain_name][
                "real_class_prediction"] = self.discriminator(images)

        for domain_name in self.options.domain_names:
            # cross-domain data
            for target_name in self.options.domain_names:
                if target_name != domain_name:
                    target_encoding = data[target_name]["encoding"]
                    cross_data = data[domain_name][target_name]
                    cross_data["random_style"] = torch.randn(
                        data[domain_name]["real"].size(0), 8).to(self.device)
                    # for mode seeking regularization
                    cross_data["random_style2"] = torch.randn(
                        data[domain_name]["real"].size(0), 8).to(self.device)

                    # decode to target domain
                    cross_data["fake"] = self.generator.decode(
                        data[domain_name]["content"],
                        data[target_name]["style"], target_encoding)
                    # decode to target domain with random style
                    cross_data["fake_random"] = self.generator.decode(
                        data[domain_name]["content"],
                        cross_data['random_style'], target_encoding)
                    cross_data["fake_random2"] = self.generator.decode(
                        data[domain_name]["content"],
                        cross_data['random_style2'], target_encoding)
                    # re-encode
                    cross_data['content_reconstruction'] = \
                        self.generator.encode_content(cross_data["fake"])
                    # re-decode
                    cross_data['cross_reconstruction'] = self.generator.decode(
                        cross_data['content_reconstruction'],
                        data[domain_name]["style_reconstruction"],
                        data[domain_name]["encoding"])
                    # for latent regression: encode random image
                    for suffix in ["", "2"]:
                        random_style_reconstruction = self.generator.\
                            encode_style(
                                cross_data[f"fake_random{suffix}"],
                                target_encoding)
                        if self.options.concat:
                            random_style_reconstruction, _ = \
                                random_style_reconstruction
                        cross_data[f"random{suffix}_style_reconstruction"] = \
                                random_style_reconstruction

                    # predictions
                    for suffix in ["", "_random", "_random2"]:
                        cross_data[f"fake{suffix}_gan_prediction"], cross_data[
                            f"fake{suffix}_class_prediction"] = \
                                self.discriminator(cross_data[f"fake{suffix}"])
        super()._generate_training_data(input_images, data)

    def _get_style(self, style):
        mean = None
        logvar = None
        if self.options.concat:
            # encoder returned mean and variance
            mean, logvar = style
            style = torch.randn(
                logvar.size(0), logvar.size(1)
                ).to(self.device).mul(logvar.mul(0.5).exp_()).add_(mean)
        return style, mean, logvar

    def _discriminator_loss(self, data):
        gan_loss = 0
        for domain, domain_data in data.items():
            classification_loss = (self.nb_domains - 1) * \
                self.classification_loss(domain_data["real_class_prediction"],
                                         domain_data["encoding"])
            for target_domain in self.options.domain_names:
                if domain != target_domain:
                    for suffix in ["", "_random", "_random2"]:
                        gan_loss += self.gan_loss.discriminator_loss(
                            domain_data["real_gan_prediction"],
                            domain_data[target_domain][
                                f"fake{suffix}_gan_prediction"])
                        if suffix:
                            classification_loss += .5 * \
                                self.classification_loss(
                                    domain_data[target_domain][
                                        f"fake{suffix}_class_prediction"],
                                    data[target_domain]["encoding"])
        return (classification_loss *
                self.options.discriminator_classification_loss_weight +
                (gan_loss / 3)) / (self.nb_domains * (self.nb_domains - 1))

    def _generator_loss(self, data):
        reconstruction_loss = 0
        content_kl_loss = 0
        style_kl_loss = 0
        gan_loss = 0
        classification_loss = 0
        cycle_reconstruction_loss = 0
        latent_regression_loss = 0
        mode_seeking_loss = 0
        for domain_name, domain_data in data.items():
            # self reconstruction loss
            reconstruction_loss += self.l1_loss(
                domain_data["real"], domain_data["reconstruction"])

            # Not present in paper, but present in implementation:
            # pseudo KL loss for content
            # Warning: though the name is kl_loss, the loss function is
            # actually of the form: mean(features^2)
            content_kl_loss += torch.mean(torch.pow(domain_data["content"], 2))

            # KL loss for style
            # Warning: though the name is kl_loss, the loss function is
            # actually of the form:
            #    _ for concat: 1 - \mu^2 - \sigma + \log(\sigma)
            #    _ otherwise: mean(features^2)
            # where \mu is the mean and \sigma the std
            if self.options.concat:
                style_kl_loss += torch.sum(
                    domain_data['mean'].pow(2).add_(domain_data['logvar'].exp())
                    .mul_(-1).add_(1).add_(domain_data['logvar'])).mul_(-0.5)
            else:
                style_kl_loss += torch.mean(torch.pow(domain_data["style"], 2))

            for target_name in self.options.domain_names:
                if target_name != domain_name:
                    for suffix in ["", "_random", "_random2"]:
                        # adversary loss for generator
                        gan_loss += self.gan_loss.generator_loss(
                            domain_data[target_name][
                                f"fake{suffix}_gan_prediction"])
                        # classification loss
                        classification_loss += self.classification_loss(
                            domain_data[target_name][
                                f"fake{suffix}_class_prediction"],
                            data[target_name]["encoding"])

                    # cycle reconstruction loss
                    cycle_reconstruction_loss += self.l1_loss(
                        domain_data["real"],
                        domain_data[target_name]["cross_reconstruction"])

                    # latent reconstruction loss
                    for suffix in ["", "2"]:
                        latent_regression_loss += self.l1_loss(
                            domain_data[target_name][
                                f"random{suffix}_style_reconstruction"],
                            domain_data[target_name][f"random_style{suffix}"])

                    mode_seeking_loss += 1 / (
                        self.l1_loss(
                            domain_data[target_name]["fake_random"],
                            domain_data[target_name]["fake_random2"]) /
                        self.l1_loss(
                            domain_data[target_name]["random_style"],
                            domain_data[target_name]["random_style2"]) + 1e-5)

        generator_loss = (
            gan_loss * self.options.gan_loss_weight +
            classification_loss *
            self.options.generator_classification_loss_weight) / 3
        generator_loss += (latent_regression_loss *
                           self.options.latent_regression_loss_weight) / 2
        generator_loss += (
            cycle_reconstruction_loss *
            self.options.cycle_reconstruction_loss_weight +
            mode_seeking_loss * self.options.mode_seeking_loss_weight)
        generator_loss /= self.nb_domains - 1
        generator_loss += (
            reconstruction_loss * self.options.reconstruction_loss_weight +
            content_kl_loss * self.options.content_kl_loss_weight +
            style_kl_loss * self.options.style_kl_loss_weight
            )
        generator_loss /= self.nb_domains
        return generator_loss

    @classmethod
    def update_arguments(cls, options):
        super().update_arguments(options)
        #### Networks parameters ####
        options.parser.add_argument(
            '--concat', action="store_true",
            help="""Use concat generator""")
        options.parser.add_argument(
            '--nb-scales', type=int, dest="nb_scales", default=1,
            help="""Number of scales for the discriminator""")
        #### Loss weights ####
        options.parser.add_argument(
            '--discriminator-classification-loss-weight', type=float,
            dest="discriminator_classification_loss_weight", default=1,
            help="Weight for discriminator classification loss")
        options.parser.add_argument(
            '--generator-classification-loss-weight', type=float, default=5,
            dest="generator_classification_loss_weight",
            help="Weight for generator classification loss")
        options.parser.add_argument(
            '--reconstruction-loss-weight', type=float, default=10,
            dest="reconstruction_loss_weight",
            help="Weight for image reconstruction loss")
        options.parser.add_argument(
            '--cycle-reconstruction-loss-weight', type=float, default=10,
            dest="cycle_reconstruction_loss_weight",
            help="Weight for cycle reconstruction loss")
        options.parser.add_argument(
            '--gan-loss-weight', type=float, dest="gan_loss_weight", default=1,
            help="Weight for GAN loss")
        options.parser.add_argument(
            '--content-kl-loss-weight', type=float, default=0.01,
            dest="content_kl_loss_weight",
            help="Weight for content KL loss")
        options.parser.add_argument(
            '--latent-regression-loss-weight', type=float, default=1,
            dest="latent_regression_loss_weight",
            help="Weight for latent regression loss")
        options.parser.add_argument(
            '--style-kl-loss-weight', type=float, default=0.01,
            dest="style_kl_loss_weight",
            help="Weight for style KL loss")
        options.parser.add_argument(
            '--mode-seeking-loss-weight', type=float, default=0.01,
            dest="mode_seeking_loss_weight",
            help="Weight for mode-seeking loss")

    def _evaluate_image(self, image, source_domain, target_domain,
                        *args, **kwargs):# pylint: disable=arguments-differ
        domain_encoding = torch.zeros((1, self.nb_domains)).to(self.device)
        domain_encoding[0, self.options.domain_names.index(source_domain)] = 1
        features, style = self.generator.encode(
            image.unsqueeze(0), domain_encoding)
        if source_domain != target_domain:
            style = Variable(torch.randn(1, 8).to(self.device))
            target_domain_encoding = torch.zeros((1, self.nb_domains)).to(
                self.device)
            target_domain_encoding[
                0, self.options.domain_names.index(target_domain)] = 1
        else:
            target_domain_encoding = domain_encoding
        return self.generator.decode(features, style, target_domain_encoding)
