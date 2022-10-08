# -*- coding: utf-8 -*-
"""Model for CartoonGAN"""
import os
import pickle
import time

import torch
from torch.nn import BatchNorm2d, BCEWithLogitsLoss, L1Loss
from torch.optim import Adam, lr_scheduler
from torchvision.models import vgg19_bn
from torchvision.utils import save_image

from ..base_model import BaseModel
from ...image_pool import ImagePool
from .discriminator import CartoonGANDiscriminator
from .generator import CartoonGANGenerator
from ..utils import get_weight_init_func, attempt_use_apex
from ..get_norm import get_norm

USE_APEX, AMP = attempt_use_apex()

class CartoonGAN(BaseModel):
    """Main class for CartoonGAN"""
    def __init__(self, options, norm=BatchNorm2d, omega=10):
        super().__init__(options)
        if options.norm is not None:
            self.norm = get_norm(options.norm)
        else:
            self.norm = norm

        # handle generator from natural images to comics
        self.generator = CartoonGANGenerator(self.norm)
        self.optimizers["generator"] = Adam(
            self.generator.parameters(), lr=options.learning_rate_generator,
            betas=(options.beta1, options.beta2))
        self.schedulers["generator"] = lr_scheduler.MultiStepLR(
            optimizer=self.optimizers["generator"],
            milestones=[options.nb_epochs // 2, options.nb_epochs // 4 * 3],
            gamma=0.1)

        # handle discriminator between natural images and comics
        self.discriminator = CartoonGANDiscriminator(self.norm)
        self.optimizers["discriminator"] = Adam(
            self.discriminator.parameters(),
            lr=options.learning_rate_discriminator,
            betas=(options.beta1, options.beta2))
        self.schedulers["discriminator"] = lr_scheduler.MultiStepLR(
            optimizer=self.optimizers["discriminator"],
            milestones=[options.nb_epochs // 2, options.nb_epochs // 4 * 3],
            gamma=0.1)

        self.trained_VGG = vgg19_bn(pretrained=True)
        # Freeze model weights
        for param in self.trained_VGG.parameters():
            param.requires_grad = False
        # Remove every layer after conv4_4
        self.trained_VGG.features = self.trained_VGG.features[:37]
        self.trained_VGG.eval()

        self.omega = (
            omega
            if options.content_loss_weight is None
            else options.content_loss_weight)

        quarter_input_size = options.input_size // 4
        self.real_label = torch.ones(
            options.batch_size, 1,
            quarter_input_size, quarter_input_size).to(self.device)
        self.fake_label = torch.zeros(
            options.batch_size, 1,
            quarter_input_size, quarter_input_size).to(self.device)

        for folder in ["Reconstruction", "Transfer"]:
            path = os.path.join(self.options.save_path, folder)
            if not os.path.isdir(path):
                os.makedirs(path)

        self.l1_loss = L1Loss().to(self.device)
        self.bce_loss = BCEWithLogitsLoss().to(self.device)

        weight_initialization_function = get_weight_init_func(options,
                                                              "gaussian")
        self.generator.apply(weight_initialization_function)
        self.discriminator.apply(weight_initialization_function)

        self.target_pool = ImagePool(options.pool_size)

        self.nets = [self.generator, self.discriminator]
        self.discriminator_loss = 0
        self.generator_loss = 0
        self.content_loss = 0

        if USE_APEX:
            [self.generator, self.discriminator], [
                self.optimizers['generator'],
                self.optimizers['discriminator']] = AMP.initialize(
                    [self.generator, self.discriminator],
                    [self.optimizers['generator'],
                     self.optimizers['discriminator']],
                    opt_level="O1",
                    num_losses=3)

    def train_epoch(self, batch_source_images, batch_target_images):# pylint: disable=arguments-differ
        """Train the CartoonGAN model for a batch of images for given source
        and target domains"""
        batch_source_images = batch_source_images[
            0]# batch contains (images, file_names)
        batch_edge_smooth_target_images = batch_target_images[0][0]
        batch_target_images = batch_target_images[1][0]
        # generate cartoonized image from real
        cartoonized_images = self.generator(batch_source_images)
        # get predicted label from discriminator for cartoonized images
        predicted_labels = self.discriminator(
            self.target_pool.query(cartoonized_images))
        # get predicted label for cartoons
        cartoon_predicted_labels = self.discriminator(batch_target_images)
        # get predicted label for edge-smooth cartoons
        edge_smooth_predicted_labels = self.discriminator(
            batch_edge_smooth_target_images)
        # compute adversary loss
        fake_loss = self.bce_loss(predicted_labels, self.fake_label)
        edge_loss = self.bce_loss(edge_smooth_predicted_labels, self.fake_label)
        real_loss = self.bce_loss(cartoon_predicted_labels, self.real_label)
        discriminator_loss = fake_loss + edge_loss + real_loss
        # compute content loss
        content_loss = self.l1_loss(self.trained_VGG(cartoonized_images),
                                    self.trained_VGG(batch_source_images))
        # compute full loss
        generator_loss = fake_loss + self.omega * content_loss

        if USE_APEX:
            with AMP.scale_loss(
                    discriminator_loss, self.optimizers["discriminator"],
                    loss_id=1) as scaled_discriminator_loss:
                scaled_discriminator_loss.backward()
        else:
            discriminator_loss.backward(retain_graph=True)
        self.optimizers["discriminator"].step()
        if USE_APEX:
            with AMP.scale_loss(
                    generator_loss, self.optimizers["generator"],
                    loss_id=2) as scaled_generator_loss:
                scaled_generator_loss.backward()
        else:
            generator_loss.backward()
        self.optimizers["generator"].step()
        if self.options.verbose:
            self.discriminator_loss += discriminator_loss.item()
            self.generator_loss += generator_loss.item()
            self.content_loss += content_loss.item()

    def log_end_epoch(self, nb_iterations):
        discriminator_loss = self.discriminator_loss / nb_iterations
        generator_loss = self.generator_loss / nb_iterations
        content_loss = self.content_loss / nb_iterations
        string = (f"Discriminator loss: {discriminator_loss:.3f}, " +
                  f"Generator loss: {generator_loss:.3f}, " +
                  f"Content loss: {content_loss:.3f}")
        self.discriminator_loss = 0
        self.generator_loss = 0
        self.content_loss = 0
        return string

    def pretrain(self, dataset):
        """Pretrain generator"""
        pre_train_hist = {}
        pre_train_hist['per_epoch_time'] = []
        pre_train_hist['reconstruction_loss'] = []
        if not self.options.resume:
            if self.options.verbose:
                print('Pre-training starts')
            start_time = time.time()
            for epoch in range(self.options.pre_train_epochs):
                epoch_start_time = time.time()
                reconstruction_losses = []
                for (_, _), ((image, _image_path), (_, _)) in dataset:
                    image = image.to(self.device)

                    # train generator G
                    self.optimizers["generator"].zero_grad()

                    x_feature = self.trained_VGG((image + 1) / 2)
                    generated_image = self.generator(image)
                    G_feature = self.trained_VGG((generated_image + 1) / 2)

                    reconstruction_loss = 10 * self.l1_loss(
                        G_feature, x_feature.detach())
                    reconstruction_losses.append(reconstruction_loss.item())
                    pre_train_hist['reconstruction_loss'].append(
                        reconstruction_loss.item())

                    if USE_APEX:
                        with AMP.scale_loss(
                                reconstruction_loss,
                                self.optimizers["generator"],
                                loss_id=3) as scaled_reconstruction_loss:
                            scaled_reconstruction_loss.backward()
                    else:
                        reconstruction_loss.backward()
                    self.optimizers["generator"].step()

                if self.options.verbose:
                    per_epoch_time = time.time() - epoch_start_time
                    pre_train_hist['per_epoch_time'].append(per_epoch_time)
                    reconstruction_loss = torch.mean(torch.FloatTensor(
                        reconstruction_losses))
                    print(
                        f"[{(epoch + 1)}/{self.options.pre_train_epochs}] - " +
                        f"time: {per_epoch_time:.2f}, Recon loss: " +
                        f"{reconstruction_loss:.3f}")

            if self.options.verbose:
                total_time = time.time() - start_time
                avg_time = sum(pre_train_hist['per_epoch_time']) / \
                    len(pre_train_hist['per_epoch_time'])
                print(
                    f"Avg one epoch time: {avg_time:.2f}, " +
                    f"total {self.options.pre_train_epochs} " +
                    f"epochs time: {total_time:.2f}")

            if self.options.save:
                with open(os.path.join(self.options.save_path,
                                       'pre_train_hist.pkl'), 'wb') as file_:
                    pickle.dump(pre_train_hist, file_)
                self.save(os.path.join(self.options.save_path,
                                       'pretrained.pth'))

            if self.options.evaluate:
                if self.options.verbose:
                    print("Evaluating pretrained generator")
                with torch.no_grad():
                    self.generator.eval()
                    for index, data in enumerate(dataset):
                        self.evaluate(index, *data, "Reconstruction")
                        if index == self.options.nb_evaluation_examples:
                            break
                    self.generator.train()
        else:
            if self.options.verbose:
                print('Load the latest generator model, no need to pre-train')

    def save(self, path, params=None):
        params = {'omega': self.omega}
        super().save(path, params)

    @classmethod
    def create_model_from_checkpoint(cls, checkpoint):
        return cls(checkpoint['options'], omega=checkpoint.get(
            'omega', getattr(checkpoint['options'], 'omega', 10)))

    @classmethod
    def update_arguments(cls, options):
        """Handle parameters for the model"""
        super().update_arguments(options)
        options.parser.add_argument(
            '--content-loss-weight', type=float, default=None,
            dest="content_loss_weight",
            help='Weight of the content loss in the generator\'s' +
            'global loss function')

    def evaluate(self, epoch_index, source_images, _target_images,#pylint: disable=arguments-differ
                 folder="Transfer"):
        for source_image, file_path in zip(*source_images):
            image = source_image.unsqueeze(0).to(self.device)
            transfered_image = self.generator(image)
            file_name = os.path.splitext(os.path.basename(file_path))[0]

            if epoch_index is not None:
                file_path = os.path.join(
                    self.options.save_path,
                    folder,
                    f"{file_name}_epoch{epoch_index}.png")
            else:
                file_path = os.path.join(
                    self.options.save_path,
                    f"{file_name}.png")

            save_image(
                [image[0], transfered_image[0]],
                file_path,
                nrow=2)
