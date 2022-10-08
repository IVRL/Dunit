"""Mixin for adding UNIT options to model"""

class UNITOptionsMixin():#pylint: disable=too-few-public-methods
    """Mixin for adding UNIT options to model"""
    @classmethod
    def update_arguments(cls, options):
        """Add parameters for the model"""
        super().update_arguments(options)
        #### Networks parameters ####
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
        options.parser.add_argument(
            '--gan-type', type=str, dest="gan_type", default='lsgan',
            choices=['lsgan', 'nsgan'],
            help="Type of GAN loss function to use")
        options.parser.add_argument(
            '--nb-scales', type=int, dest="nb_scales", default=3,
            help="Number of scales at which to apply the discriminator")
        options.parser.add_argument(
            '--nb-layers-discriminator', type=int, dest="nb_layers", default=4,
            help="Number of layers in the discriminator")
        #### Loss weights ####
        options.parser.add_argument(
            '--gan-loss-weight', type=float, dest="gan_loss_weight", default=1,
            help="Weight for GAN loss")
        options.parser.add_argument(
            '--cycle-encoder-loss-weight', type=float, default=0.01,
            dest="cycle_encoder_loss_weight",
            help="Weight for cycle encoder loss")
        options.parser.add_argument(
            '--cycle-consistency-loss-weight', type=float, default=10,
            dest="cycle_consistency_loss",
            help="Weight for cycle consistency loss")
