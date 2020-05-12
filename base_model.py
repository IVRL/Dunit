# -*- coding: utf-8 -*-
"""Base class for models"""
import os
import torch
from ..with_device import WithDeviceMixin
from .utils import get_optimizer, get_scheduler, get_weight_init_func, \
    attempt_use_apex

USE_APEX, AMP = attempt_use_apex()

class BaseModel(WithDeviceMixin):
    """Base class for models"""
    schedulers = {}
    optimizers = {}
    generators = None
    discriminators = None

    def __init__(self, options):
        super().__init__()
        self.options = options
        self.metric = 0  # used for learning rate policy 'plateau'
        self.nets = []

        self.discriminator_loss = 0
        self.generator_loss = 0

    def _end_setup(self, use_lists=False):
        # create optimizers and schedulers
        self.create_optimizers(use_lists)

        for name in self.optimizers.keys():# pylint: disable=consider-iterating-dictionary
            # create schedulers
            self.schedulers[name] = get_scheduler(
                self.optimizers[name], self.options)

        self._folder_creation()
        self._set_nets(use_lists)
        # move nets to device
        for net in self.nets:
            net.to(self.device)
        self._weight_initialization()
        self._initialize_apex(use_lists)

    def _initialize_apex(self, use_lists):
        # use apex optimizations
        if USE_APEX and self.options.use_mixed_precision:
            self.nets, [
                self.optimizers["generators"],
                self.optimizers["discriminators"],
                ] = AMP.initialize(
                    self.nets,
                    [self.optimizers["generators"],
                     self.optimizers["discriminators"],],
                    opt_level="O1", num_losses=2)
            i = 0
            for dictionary in [self.generators, self.discriminators]:
                for key in (range(len(dictionary)) if use_lists
                            else dictionary.keys()):
                    dictionary[key] = self.nets[i]
                    i += 1

    def create_optimizers(self, use_lists):
        """Create optimizers"""
        for name, nets in [("generator", self.generators),
                           ("discriminator", self.discriminators)]:
            plural_name = f"{name}s"
            self._create_optimizer(nets, use_lists, plural_name, name)

    def _create_optimizer(self, nets, use_lists, name, role_name=None):
        self.optimizers[name] = get_optimizer(
            self.options,
            [param
             for net in (nets if use_lists else nets.values())
             for param in net.parameters() if param.requires_grad],
            net_role=role_name if role_name is not None else name)

    def _set_nets(self, use_lists):
        if use_lists:
            self.nets = self.generators + self.discriminators
        else:
            self.nets = list(self.generators.values()) + list(
                self.discriminators.values())

    def _weight_initialization(self):
        """Define weight initialization"""
        for net in self.nets:
            net.apply(get_weight_init_func(self.options))

    def _folder_creation(self):
        """Create folder architecture for saving images"""
        for folder_name in ['from_source', 'from_target']:
            folder_path = os.path.join(self.options.save_path, folder_name)
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

    def _loss_backward(self, loss, optimizer_name, loss_id, retain_graph=False):
        if USE_APEX and self.options.use_mixed_precision:
            with AMP.scale_loss(
                    loss, self.optimizers[optimizer_name],
                    loss_id=loss_id) as scaled_loss:
                scaled_loss.backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)
        self.optimizers[optimizer_name].step()

    def pretrain(self, dataset):
        """Pre-train the model if needed

        Arguments
        ---------
        dataset
            Dataset to use for pre-training
        """

    def train_epoch(self, *args, **kwargs):
        """Train the model"""
        raise NotImplementedError(
            "The `train` method is not implemented for this model")

    def evaluate(self, *args, **kwargs):
        """Save an example of image transfered by the model"""
        raise NotImplementedError(
            "The `evaluate` method is not implemented for this model")

    def update_learning_rates(self):
        """Update learning rates at the end of an epoch"""
        for scheduler in self.schedulers.values():
            #if self.options.learning_rate_policy == 'plateau':
            #    scheduler.step(self.metric)
            #else:
            scheduler.step()

    def __str__(self):
        nb_params = 0
        string = ""
        for net in self.nets:
            nb_params += net.get_number_params()
            string = f"{string}{net}\n"

        string += f"Total number of parameters for model: {nb_params}"
        return string

    def log_end_epoch(self, nb_iterations):
        """String to display at the end of an epoch"""
        discriminator_loss = self.discriminator_loss / nb_iterations
        generator_loss = self.generator_loss / nb_iterations
        string = (f"Discriminator loss: {discriminator_loss:.3f}, " +
                  f"Generator loss: {generator_loss:.3f}")
        self.discriminator_loss = 0
        self.generator_loss = 0
        return string

    def save(self, path, params=None):
        """Save the model for later re-use"""
        if params is None:
            params = {}

        params['options'] = self.options
        params['metric'] = self.metric
        params['nets'] = [net.state_dict() for net in self.nets]
        params['optimizers'] = {key: optimizer.state_dict()
                                for key, optimizer in self.optimizers.items()}
        torch.save(params, path)

    @classmethod
    def load(cls, path, options, device):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=device)
        options.merge(checkpoint['options'])
        model = cls.create_model_from_checkpoint(checkpoint)
        model.metric = checkpoint['metric']
        for key, state_dict in checkpoint['optimizers'].items():
            model.optimizers[key].load_state_dict(state_dict)

        for index, state_dict in enumerate(checkpoint['nets']):
            model.nets[index].load_state_dict(state_dict)
            model.nets[index].to(device)

        return model

    @classmethod
    def create_model_from_checkpoint(cls, checkpoint):
        """Create a model using the parameters stored in the checkpoint"""
        return cls(checkpoint['options'])

    def train(self):
        """Call train on all nets"""
        for net in self.nets:
            net.train()

    def eval(self):
        """Call eval an all nets"""
        for net in self.nets:
            net.eval()

    @classmethod
    def update_arguments(cls, options):
        """Add parameters for the model"""
        options.parser.add_argument(
            '--norm', type=str, default=None,
            dest="norm", choices=['batch', 'instance'],
            help='Norm to use in the networks')

    #### Auxiliary methods ####
    # to be implemented in child classes and mixins
    def _generate_training_data(self, input_images, data):
        pass

    def _generator_loss(self, data):
        pass

    def _discriminator_loss(self, data):
        pass

    def _additional_training(self, data):
        pass
