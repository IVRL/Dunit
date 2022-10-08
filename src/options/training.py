# -*- coding: utf-8 -*-
"""Class for registering training args"""
from .base import BaseOptions
from ..models.utils import AVAILABLE_SCHEDULERS, AVAILABLE_OPTIMIZERS, \
    AVAILABLE_INITILIZATIONS

class TrainingOptions(BaseOptions):
    """Class defining training-specific options"""

    def __init__(self):
        super().__init__()
        #### General arguments ####
        self.parser.add_argument(
            '--nb-epochs', '-e', type=int, default=100,
            dest="nb_epochs",
            help='Number of epochs to train')
        self.parser.add_argument(
            '--nb-pretraining-epochs', type=int, default=10,
            dest="pre_train_epochs",
            help='Number of epochs to pretrain model')
        self.parser.add_argument(
            '--evaluate', action='store_true',
            help='Evaluate examples during training')
        self.parser.add_argument(
            '--nb-evaluation-examples', type=int, default=4,
            dest='nb_evaluation_examples',
            help='Maximum number of examples to evaluate')
        self.parser.add_argument(
            '--evaluation-frequency', type=int, default=0,
            dest='evaluation_frequency',
            help='Number of epochs between two evaluations')
        self.parser.add_argument(
            '--save', action='store_true',
            help='Save intermediary results and models')
        self.parser.add_argument(
            '--save-path', type=str, dest="save_path",
            help='Path where to save intermediary results and models')
        self.parser.add_argument(
            '--save-frequency', type=int, dest="save_frequency", default=0,
            help='Number of epochs between two checkpoints')
        ### Scheduler-related arguments ####
        self.parser.add_argument(
            '--scheduler', type=str, dest="scheduler_type",
            choices=AVAILABLE_SCHEDULERS.keys(), default="StepLR",
            help="Scheduler to use for optimization")
        self.parser.add_argument(
            '--scheduler-step-size', type=int, default=50, dest="step_size",
            help="Step size for scheduler")
        self.parser.add_argument(
            '--gamma', type=float, default=0.1,
            help="Gamma for scheduler")
        self.parser.add_argument(
            '--last-epoch-scheduler', dest="iterations", type=int, default=-1,
            help="Index of the last epoch for which to use the scheduler")
        self.parser.add_argument(
            '--milestones', dest="milestones", type=list, default=[30, 50],
            help="List of epoch indexes on which to apply the weight" +
            "decay to the learning rate")
        #### Optimization related arguments ####
        self.parser.add_argument(
            '--optimizer', type=str, dest="optimizer_type", default="Adam",
            choices=AVAILABLE_OPTIMIZERS.keys(),
            help="Optimizer")
        self.parser.add_argument(
            '--learning-rate-generator', '-lrG', type=float, default=0.0002,
            dest='learning_rate_generator',
            help="Learning rate for the generator")
        self.parser.add_argument(
            '--learning-rate-discriminator', '-lrD', type=float, default=0.0002,
            dest='learning_rate_discriminator',
            help="Learning rate for the discriminator")
        self.parser.add_argument(
            '--beta1', type=float, default=0.5,
            help='Beta1 for Adam optimizer')
        self.parser.add_argument(
            '--beta2', type=float, default=0.999,
            help='Beta2 for Adam optimizer')
        self.parser.add_argument(
            '--weight-decay', type=float, default=0, dest="weight_decay",
            help='Weight decay for Adam optimizer')
        #### Weight initialization arguments ####
        self.parser.add_argument(
            '--initialization', type=str, default="gaussian",
            choices=AVAILABLE_INITILIZATIONS.keys(),
            help="Method for weight initialization")
        self.parser.add_argument(
            '--initial-mean', type=float, dest="initial_mean", default=0.0,
            help="Mean for gaussian weight initialization")
        self.parser.add_argument(
            '--initial-std', type=float, dest="initial_std", default=0.02,
            help="Std for gaussian weight initialization")
        self.parser.add_argument(
            '--gain', type=float, default=0.02,
            help="Gain for weight initialization")
        self.parser.add_argument(
            '--kaiming-nonlinearity', type=str, default="leaky_relu",
            dest="kaiming_nonlinearity",
            help="Gain for weight initialization")
        self.parser.add_argument(
            '--kaiming-slope', type=float, default=0.0, dest="kaiming_slope",
            help="Slope for Kaiming weight initialization")
        self.parser.add_argument(
            '--kaiming-mode', type=str, default='fan_in', dest="kaiming_mode",
            choices=['fan_in', 'fan_out'],
            help="Mode for Kaiming weight initialization")
        self.parser.add_argument(
            '--initial-constant', type=float, default=0.1,
            dest="initial_constant_weight",
            help="Initial constant weight for weight initialization")

    def preprocess_args(self):
        super().preprocess_args()
        if self.options.save and self.options.save_path is None:
            self.options.save_path = f"{self.options.name}_results"
