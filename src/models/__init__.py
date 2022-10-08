# -*- coding: utf-8 -*-
"""Module containing all models"""
import os
import torch

from .aug_gan import AugGAN
from .train_model import train_model # for the sake of reusing it higher-up
from .evaluate_model import evaluate_model
from .cartoon_gan import CartoonGAN
from .cycle_gan import CycleGAN
from .custom_gan import CustomGAN
from .drit import DRIT
from .unit import UNIT
from .munit import MUNIT
from .object_detection import DRITObjectDetection, UNITObjectDetection

def get_model_path(options):
    """Retrieve path to stored model"""
    folder_path = options.options.checkpoint_path \
        if options.options.checkpoint_path is not None \
        else options.options.save_path
    if options.options.checkpoint_index is not None:
        try:
            starting_epoch = int(options.options.checkpoint_index)
            path = os.path.join(folder_path, f"epoch_{starting_epoch}.pth")
            setattr(options.options, "starting_epoch", starting_epoch)
        except ValueError:
            path = os.path.join(folder_path,
                                f"{options.options.checkpoint_index}.pth")
    else:
        indexes = sorted([
            int(file_name.replace('epoch_', ''))
            for file_name in [
                os.path.splitext(os.path.basename(file_name))[0]
                for file_name in os.listdir(folder_path)
                if file_name.endswith('.pth') and
                file_name.startswith("epoch")]])
        if not indexes:
            indexes = sorted([
                os.path.basename(file_name)
                for file_name in os.listdir(folder_path)
                if file_name.endswith('.pth')])
            path = os.path.join(folder_path, indexes[0])
        else:
            path = os.path.join(folder_path, f"epoch_{indexes[-1]}.pth")
            setattr(options.options, "starting_epoch", indexes[-1])
    return path

AVAILABLE_MODELS = {
    "CartoonGAN": CartoonGAN,
    "CycleGAN": CycleGAN,
    "DRIT": DRIT,
    "UNIT": UNIT,
    "MUNIT": MUNIT,
    "AugGAN": AugGAN,
    "Custom": CustomGAN,
    "DRITObjectDetection": DRITObjectDetection,
    "UNITObjectDetection": UNITObjectDetection,
    }

def create_model(options):
    """Create a model from the given options"""
    model_class = AVAILABLE_MODELS[options.options.model]
    if options.options.resume:
        path = get_model_path(options)
        return model_class.load(
            path=path,
            options=options,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    return model_class(options.options)

def load_model(options):
    """Load model according to the options"""
    path = get_model_path(options)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device)
    options.merge(checkpoint['options'])
    model_class = AVAILABLE_MODELS[checkpoint['options'].model]
    model = model_class.create_model_from_checkpoint(checkpoint)

    for index, state_dict in enumerate(checkpoint['nets']):
        model.nets[index].load_state_dict(state_dict)
        model.nets[index].to(device)
    return model
