# -*- coding: utf-8 -*-
"""Utilities for models"""
from torch.nn.init import constant_, kaiming_normal_, normal_, orthogonal_, \
    xavier_normal_
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR

AVAILABLE_OPTIMIZERS = {
    "Adam": lambda options, parameters, net_role: Adam(
        parameters,
        lr=getattr(options, f"learning_rate_{net_role}"),
        betas=(options.beta1, options.beta2),
        weight_decay=options.weight_decay),
    }

def get_optimizer(options, parameters, net_role):
    """Create and return an optimizer"""
    return AVAILABLE_OPTIMIZERS[options.optimizer_type](
        options, parameters, net_role)

AVAILABLE_SCHEDULERS = {
    "StepLR": lambda optimizer, options: StepLR(
        optimizer, step_size=options.step_size, gamma=options.gamma,
        last_epoch=options.iterations),
    "Constant": lambda *args: None,
    "MultiStepLR": lambda optimizer, options: MultiStepLR(
        optimizer=optimizer, gamma=options.gamma,
        milestones=options.milestones),
    }

def get_scheduler(optimizer, options):
    """Create and return scheduler"""
    return AVAILABLE_SCHEDULERS[options.scheduler_type](optimizer, options)

AVAILABLE_INITILIZATIONS = {
    "gaussian": lambda weights, options: normal_(
        weights, options.initial_mean, options.initial_std),
    "xavier": lambda weights, options: xavier_normal_(
        weights, gain=options.gain),
    "orthogonal": lambda weights, options: orthogonal_(
        weights, gain=options.gain),
    "default": lambda *args: None,
    "kaiming": lambda weights, options: kaiming_normal_(
        weights, a=options.kaiming_slope, mode=options.kaiming_mode,
        nonlinearity=options.kaiming_nonlinearity),
    "constant": lambda weights, options: constant_(
        weights, options.initial_constant_weight)
    }

def get_weight_init_func(options, type_=None):
    """Retrieve function to apply for weight initialization"""
    def weights_initialization(module):
        """Function initializing the weights of a module"""
        classname = module.__class__.__name__
        if ('Conv' in classname or 'Linear' in classname) and \
                hasattr(module, 'weight'):
            AVAILABLE_INITILIZATIONS[
                type_ if type_ is not None else options.initialization
                ](module.weight, options)

        if hasattr(module, 'bias') and module.bias is not None:
            constant_(module.bias, 0.0)

    return weights_initialization

def attempt_use_apex():
    """Try to import apex for optimization purposes"""
    try:
        from apex import amp#pylint: disable=import-outside-toplevel
        return True, amp
    except ImportError:
        print("Apex library not installed, thus not used")
        return False, None
