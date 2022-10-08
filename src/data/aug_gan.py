"""Module creating a dataset with segmentation for each image
if it does not exist yet"""
import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models.segmentation import deeplabv3_resnet101
from .source_target import SourceTargetDataset

def decode_segmap(image, nb_classes=21):
    """Turn segmentation map to image"""
    label_colors = np.array([
        (0, 0, 0),  # 0=background
        # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
        # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
        (192, 128, 128),
        # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    red = np.zeros_like(image).astype(np.uint8)#pylint: disable=no-member
    green = np.zeros_like(image).astype(np.uint8)#pylint: disable=no-member
    blue = np.zeros_like(image).astype(np.uint8)#pylint: disable=no-member

    for label in range(0, nb_classes):
        idx = image == label
        red[idx] = label_colors[label, 0]
        green[idx] = label_colors[label, 1]
        blue[idx] = label_colors[label, 2]

    rgb = np.stack([red, green, blue], axis=2)
    return rgb


def add_segmentation_annotation(options, domain):
    """Use pretrained net to create segmentation data"""
    if getattr(options, f"{domain}_annotation", None) is None:
        # load segmentation model
        model = deeplabv3_resnet101(pretrained=True)
        # create annotations, and store them
        transform = Compose([
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
        root, _, file_names = next(os.walk(
            os.path.join(options.dataroot, getattr(options, domain))))
        for file_name in file_names:
            img = Image.open(os.path.join(root, file_name))
            print(img.size)
            inp = transform(img).unsqueeze(0)
            print(inp.shape)
            out = model(inp)
            print(out.shape)
            out = out['out']
            omn = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
            rgb = decode_segmap(omn)
            # save image
            Image.fromarray(rgb).save(
                os.path.join(
                    getattr(options, domain), 'segmentation', file_name))

            # add path to created annotations
            setattr(options, f"{domain}_annotation",
                    os.path.join(getattr(options, domain), 'segmentation'))

def create_aug_gan_dataset(options):
    """Create dataset with segmentation ground-truth for AugGAN"""
    if options.verbose:
        print("Creating dataset with segmentation")

    if options.source_annotation is None:
        add_segmentation_annotation(options, "source")
    if options.target_annotation is None:
        add_segmentation_annotation(options, "target")

    return SourceTargetDataset(options, with_annotations=True)
