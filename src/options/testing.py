# -*- coding: utf-8 -*-
"""Class for registering testing args"""
import os

from .base import BaseOptions

class TestingOptions(BaseOptions):
    """Class defining testing-specific options"""

    def __init__(self):
        super().__init__()
        #### General arguments ####
        self.parser.add_argument(
            '--save-path', type=str, dest="save_path",
            help='Path where to save images')
        self.parser.add_argument(
            '--direction', type=str, default="both",
            choices=['both', 'to_source', 'to_target'],
            help="Direction for image transfer")

    def preprocess_args(self):
        super().preprocess_args()
        self.options.resume = True
        if self.options.model == "CartoonGAN":
            self.options.direction = 'to_target'
        if self.options.direction != 'both':
            self.options.dataset_type = "ImageFolder"

        if self.options.checkpoint_path is None:
            self.options.checkpoint_path = os.path.join(
                f"{self.options.name}_results")
        if self.options.save_path is None:
            self.options.save_path = os.path.join(
                f"{self.options.name}_results", 'test')
