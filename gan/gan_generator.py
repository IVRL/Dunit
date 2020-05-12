"""Generator using AugGAN structure"""
from torch.nn import InstanceNorm2d, ReflectionPad2d
from ..base_module import BaseModule
from ..aug_gan.encoder import Encoder
from ..aug_gan.multi_task_generator import MultiTaskGenerator

class AugGANGenerator(BaseModule):
    """Generator using AugGAN structure"""
    def __init__(self, options):
        super().__init__(options=options)
        self.encoder = Encoder(
            self.options.nb_channels, self.options.last_layer_size,
            nb_blocks=self.options.nb_blocks, norm=InstanceNorm2d,
            padding_module=ReflectionPad2d).to(self.device)

        self.decoder = MultiTaskGenerator(
            4 * self.options.last_layer_size, self.options.nb_channels,
            parse_nc=32, norm=InstanceNorm2d, nb_blocks=6,
            padding_module=ReflectionPad2d).to(self.device)

    def encode(self, images):
        """Encode images to latent space"""
        return self.encoder(images)

    def decode(self, features):
        """Decode features to images"""
        return self.decoder(features)
