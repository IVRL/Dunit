# -*- coding: utf-8 -*-
"""Base class for registering options"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ..models import AVAILABLE_MODELS
from ..data import AVAILABLE_DATASETS, AVAILABLE_ANNOTATED_DATASETS

class BaseOptions():
    """Base class for registering options"""
    options = None

    def __init__(self):
        self.parser = ArgumentParser(
            description="Basic options for models",
            formatter_class=ArgumentDefaultsHelpFormatter)
        #### File related arguments ####
        self.parser.add_argument(
            '--resume', '-r', action='store_true', dest="resume",
            help="Load saved model")
        self.parser.add_argument(
            '--checkpoint-path', type=str, dest="checkpoint_path",
            help="Path to the checkpoint to load")
        self.parser.add_argument(
            '--checkpoint-index', type=int, dest="checkpoint_index",
            help="""Index of the checkpoint to load.
Equivalent to the number of epochs on which the model has already been trained
""")
        #### Model related arguments ####
        self.parser.add_argument(
            '--model', '-m', type=str, default='CartoonGAN',
            dest="model", choices=AVAILABLE_MODELS.keys(),
            help='Chooses which model to use.')
        #### Dataset related arguments ####
        self.parser.add_argument(
            '--dataset-type', type=str, default='SourceTarget',
            dest="dataset_type", choices=AVAILABLE_DATASETS.keys(),
            help='Chooses which type of dataset to use')
        self.parser.add_argument(
            '--data-root', type=str, default='data',
            dest="dataroot", help='Give the root folder for data.')
        self.parser.add_argument(
            '--source', '-s', type=str, default="source",
            dest="source", help="Give the folder for source images")
        self.parser.add_argument(
            '--target', '-t', type=str, default="target",
            dest="target", help="Give the folder for target images")
        self.parser.add_argument(
            '--domain-names', '-d', type=str, default=None,
            dest="domain_names", help="Give the names for the domains")
        self.parser.add_argument(
            '--domain-folders', type=str, default=None,
            dest="domain_folders", help="Give the folders for the domains")
        self.parser.add_argument(
            '--domain-annotations', type=str, default=None,
            dest="domain_annotations",
            help="Give the annotation types for the domains")
        self.parser.add_argument(
            '--domain-annotation-folders', type=str, default=None,
            dest="domain_annotation_folders",
            help="Give the folders containing the annotations for the domains")
        self.parser.add_argument(
            '--batch-size', type=int, default=1,
            dest="batch_size", help="Size of batches for data")
        self.parser.add_argument(
            '--serial-batches', action='store_true',
            dest="serial_batches",
            help='If true, takes images in order to make batches,' +
            'otherwise takes them randomly')
        self.parser.add_argument(
            '--num-threads', default=1, type=int,
            dest="num_threads",
            help='Number of threads for loading data')
        self.parser.add_argument(
            '--max-dataset-size', type=int, default=float("inf"),
            dest="max_dataset_size",
            help="""Maximum number of samples allowed per dataset.
If the dataset directory contains more than max_dataset_size,
only a subset is loaded.""")
        self.parser.add_argument(
            '--recursive', action="store_true",
            help="""Set to load data recursively.""")
        self.parser.add_argument(
            '--with-annotations', action='store_true',
            dest="with_annotations",
            help='If true, add annotations to the images')
        self.parser.add_argument(
            '--source-annotation', type=str, default=None,
            dest="source_annotation",
            help="Give the folder for annotations of source images")
        self.parser.add_argument(
            '--source-annotation-type', type=str, default="segmentation",
            dest="source_annotation_type",
            choices=AVAILABLE_ANNOTATED_DATASETS.keys(),
            help="Give the type of annotations for source images")
        self.parser.add_argument(
            '--target-annotation', type=str, default=None,
            dest="target_annotation",
            help="Give the folder for annotations of target images")
        self.parser.add_argument(
            '--target-annotation-type', type=str, default="segmentation",
            dest="target_annotation_type",
            choices=AVAILABLE_ANNOTATED_DATASETS.keys(),
            help="Give the type of annotations for target images")
        self.parser.add_argument(
            "--pool-size", type=int, default=50, dest="pool_size",
            help="Maximum size of the image pool")

        #### Image transformation arguments ####
        self.parser.add_argument(
            '--force-image-preprocessing', action='store_true',
            dest='force_image_preprocessing',
            help="""Force image preprocessing""")
        self.parser.add_argument(
            '--input-size', default=256, dest='input_size', type=int,
            help="""Size of the input for the model""")
        self.parser.add_argument(
            '--center-crop', dest='center_crop', action='store_true',
            help="Crop images around the center")
        self.parser.add_argument(
            '--crop', dest='crop', action='store_true',
            help="Crop images randomly")
        self.parser.add_argument(
            '--crop-height', default=None, dest='crop_height', type=int,
            help="""Cropping height for images""")
        self.parser.add_argument(
            '--crop-width', default=None, dest='crop_width', type=int,
            help="""Cropping width for images""")
        self.parser.add_argument(
            '--crop-size', default=256, dest='crop_size', type=int,
            help="""Cropping size for images""")
        self.parser.add_argument(
            '--flip-horizontal', dest='flip_horizontal', action='store_true',
            help="Flip images horizontally randomly")
        self.parser.add_argument(
            '--flip-vertical', dest='flip_vertical', action='store_true',
            help="Flip images vertically randomly")
        self.parser.add_argument(
            '--proba-vertical-flip', dest='proba_vertical_flip', type=float,
            help="Probability of flipping an image vertically", default=0.5)
        self.parser.add_argument(
            '--proba-horizontal-flip', dest='proba_horizontal_flip',
            type=float,
            help="Probability of flipping an image horizontally", default=0.5)
        self.parser.add_argument(
            '--normalize', action="store_true",
            help="Normalize images")
        self.parser.add_argument(
            '--normalize-mean', dest="normalize_mean",
            type=tuple, default=(0.5, 0.5, 0.5),
            help="Mean to use for each channel for normalization")
        self.parser.add_argument(
            '--normalize-std', dest="normalize_std",
            type=tuple, default=(0.5, 0.5, 0.5),
            help="Std to use for each channel for normalization")
        #### Various parameters ####
        self.parser.add_argument(
            '--verbose', '-v', action="store_true",
            dest="verbose",
            help="If specified, print more debug information")
        self.parser.add_argument(
            '--name', '-n', default="random", dest='name', type=str,
            help="""Name to give to the model""")
        self.parser.add_argument(
            '--use-mixed-precision', '-o', action="store_true",
            dest='use_mixed_precision',
            help="""Set to attempt using mixed precision""")

    def parse(self):
        """Parse command-line arguments"""
        self.options, _ = self.parser.parse_known_args()
        self.update_arguments()
        self.options = self.parser.parse_args()
        self.preprocess_args()
        if self.options.verbose:
            print("---------------Options-------------")
            for key, value in sorted(vars(self.options).items()):
                comment = ''
                default = self.parser.get_default(key)
                if value != default:
                    comment = f"\t[default: {default}]"
                print(f"{key:<55}: {value!s:>10}{comment}")
            print("--------------End options----------")
        return self.options

    def merge(self, options):
        """Merge current options with options from stored model"""
        for option_name, option_value in vars(self.options).items():
            if option_name not in [
                    "beta1", "beta2", "initial_mean", "initial_std"]:
                setattr(options, option_name, option_value)

    def preprocess_args(self):
        """Preprocess arguments"""
        if self.options.resume:
            try:
                setattr(self.options, "starting_epoch",
                        int(self.options.checkpoint_index))
            except TypeError:
                setattr(self.options, "starting_epoch", 0)
        else:
            setattr(self.options, "starting_epoch", 0)

        if (self.options.crop or self.options.center_crop) and \
                (self.options.crop_height is None or
                 self.options.crop_width is None):
            self.options.crop_height = self.options.crop_size
            self.options.crop_width = self.options.crop_size

        # handle args for multiple domains
        if self.options.domain_names is not None:
            self.options.domain_names = [
                name if name != "None" or not name.strip() else None
                for name in self.options.domain_names.split(',')]
            self.options.domain_folders = [
                name if name != "None" or not name.strip() else None
                for name in self.options.domain_folders.split(',')]
            assert len(self.options.domain_folders) == len(
                self.options.domain_names), \
                f"{len(self.options.domain_folders)} != " + \
                f"{len(self.options.domain_names)}"
            if self.options.with_annotations:
                self.options.domain_annotations = self.options.\
                    domain_annotations.split(',')
                self.options.domain_annotation_folders = [
                    name if name != "None" or not name.strip() else None
                    for name in self.options.domain_annotation_folders
                    .split(',')]
                assert len(self.options.domain_annotations) == len(
                    self.options.domain_names)
                assert len(self.options.domain_annotation_folders) == len(
                    self.options.domain_names)
        else:
            self.options.domain_names = ["source", "target"]
            self.options.domain_folders = ["source", "target"]

    def update_arguments(self):
        """Update arguments based on model"""
        if self.options.model is not None:
            AVAILABLE_MODELS[self.options.model].update_arguments(self)
