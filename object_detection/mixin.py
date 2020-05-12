"""Mixin for object detection"""
import os
import json
import pickle
import time
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from ..losses import IoULoss, AVAILABLE_IOU
from ..utils import attempt_use_apex

USE_APEX, AMP = attempt_use_apex()

class ObjectDetectionModelMixin():
    """Mixin for adding an object detection loss to an object detection model"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_detection_loss = IoULoss(
            iou_func=AVAILABLE_IOU[self.options.iou_func],
            reduction=self.options.loss_reduction,
            area_normalization=self.options.area_normalization).to(self.device)

    def _get_loss_dict(self, loss_dict, predictions, ground_truth):
        loss_dict["object_detection_loss"] = self.object_detection_loss(
            predictions, ground_truth)
        return loss_dict

    @classmethod
    def update_arguments(cls, options):
        """Update arguments"""
        super().update_arguments(options)
        #### IoU loss parameters ####
        options.parser.add_argument(
            '--iou-func', type=str, choices=AVAILABLE_IOU.keys(),
            default="simple", dest="iou_func",
            help="Function to use when computing the IoU of two boxes")
        options.parser.add_argument(
            '--loss-reduction', type=str, choices=['mean', 'sum'],
            default="mean", dest="loss_reduction",
            help="Reduction to use on batches for the IoU loss")
        options.parser.add_argument(
            '--area-normalization', action="store_true",
            dest="area_normalization",
            help="Set to normalize the IoU by the area of the boxes")

class ObjectDetectionMixin(ObjectDetectionModelMixin):
    """Mixin for object detection"""
    def _end_setup(self, use_lists=False):
        self.detection_net = fasterrcnn_resnet50_fpn(
            pretrained=False, pretrained_backbone=False,
            num_classes=self.options.num_classes).to(self.device)
        self.iou_loss = 0

        self._create_optimizer([self.detection_net], True, "detection")
        super()._end_setup(use_lists)

    def pretrain(self, dataset):
        """Pretrain faster RCNN on the source domain"""
        self.detection_net.train()
        pre_train_hist = {}
        pre_train_hist['per_epoch_time'] = []
        if not self.options.resume:
            if self.options.verbose:
                print('Object detection net training starts')
            start_time = time.time()
            for epoch in range(self.options.pre_train_epochs):
                epoch_start_time = time.time()
                losses = []
                for domain_data, *_ in dataset:
                    if isinstance(domain_data[0], dict):
                        data, image_paths = domain_data
                    else:
                        _, data, image_paths = domain_data
                    image = data["image"]
                    bboxes = data["boxes"]
                    boxes = []
                    for index in range(len(image_paths)):
                        boxes.append({
                            "boxes": bboxes["boxes"][index],
                            "labels": bboxes["labels"][index]})


                    # train detection network
                    self.optimizers["detection"].zero_grad()
                    loss_dict = self.detection_net(image, boxes)
                    loss = sum(loss for loss in loss_dict.values()
                               if not torch.isnan(loss))
                    self._loss_backward(loss, "detection", 1)
                    losses.append(loss)

                self.schedulers["detection"].step()

                if self.options.verbose:
                    per_epoch_time = time.time() - epoch_start_time
                    pre_train_hist['per_epoch_time'].append(per_epoch_time)
                    loss = torch.mean(torch.FloatTensor(
                        losses)).item()
                    print(
                        f"[{(epoch + 1)}/{self.options.pre_train_epochs}] - " +
                        f"time: {per_epoch_time:.2f}, Training loss: " +
                        f"{loss:.3f}")

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
        else:
            if self.options.verbose:
                print('Load already trained model, no need to re-train')
        self.detection_net.eval()

    def _initialize_apex(self, use_lists):
        super()._initialize_apex(use_lists)
        if USE_APEX and self.options.use_mixed_precision:
            self.detection_net, self.optimizers["detection"] = AMP.initialize(
                self.detection_net, self.optimizers["detection"],
                opt_level="O1", num_losses=1)

    def log_end_epoch(self, nb_iterations):
        """String to display at the end of an epoch"""
        string = super().log_end_epoch(nb_iterations)
        iou_loss = self.iou_loss / nb_iterations
        string += f", IoU loss: {iou_loss:.3f}"
        self.iou_loss = 0
        return string

    def _generate_training_images(self, source_images, target_images, images):
        """Used for bidirectional models"""
        images["source"]["boxes"] = source_images[0]["boxes"]
        images["target"]["boxes"] = target_images[0]["boxes"]

        super()._generate_training_images(source_images, target_images, images)

        for domain in ["source", "target"]:
            if images[domain]["boxes"] is not None:
                boxes = []
                for index in range(images[domain]['real'].size(0)):
                    boxes.append({
                        "boxes": images[domain]["boxes"]["boxes"][index],
                        "labels": images[domain]["boxes"]["labels"][
                            index]})
                images[domain]["boxes"] = boxes
            # if wanted/needed, get pseudo ground truth
            if self.options.always_use_pseudo_ground_truth or (
                    self.options.use_pseudo_ground_truth_if_unavailable and
                    images[domain]["boxes"] is None):
                images[domain]["boxes"] = self.detection_net(
                    images[domain]["real"])

            # compute object detection for generated images
            if images[domain]["boxes"] is not None:
                images[domain]["fake_boxes"] = self.detection_net(
                    images[self._negate_direction(domain)]["fake"])

                # compute object detection for reconstructed images
                if self.options.cycle_object_detection_loss_weight > 0:
                    images[domain]["reconstruction_boxes"] = \
                        self.detection_net(images[domain]["reconstruction"])

    def _generate_training_data(self, input_images, data):
        """Used for multi-modal models"""

        super()._generate_training_data(input_images, data)

        for domain, domain_data, _ in input_images:
            domain = domain[0]
            data[domain]["boxes"] = domain_data.get("boxes", None)
            if data[domain]['boxes']:
                boxes = []
                for index in range(data[domain]['real'].size(0)):
                    boxes.append({
                        "boxes": data[domain]["boxes"]["boxes"][index],
                        "labels": data[domain]["boxes"]["labels"][index]
                        })
                data[domain]["boxes"] = boxes
            # if wanted/needed, get pseudo ground truth
            if self.options.always_use_pseudo_ground_truth or (
                    self.options.use_pseudo_ground_truth_if_unavailable and
                    data[domain]["boxes"] is None):
                data[domain]["boxes"] = self.detection_net(
                    data[domain]["real"])

            if data[domain]["boxes"] is not None:
                for target_domain in self.options.domain_names:
                    if target_domain != domain:
                        # compute object detection for generated images
                        data[domain][target_domain]["boxes"] = \
                            self.detection_net(
                                data[domain][target_domain]["fake"])

                # compute object detection for reconstructed images
                if self.options.cycle_object_detection_loss_weight > 0:
                    data[domain]["reconstruction_boxes"] = \
                        self.detection_net(data[domain]["reconstruction"])

    def _object_detection_loss(self, images, domain):
        if self.multimodal:
            loss = 0
            if images[domain]["boxes"] is not None:
                for target_name in self.options.domain_names:
                    if target_name != domain:
                        loss += self.object_detection_loss(
                            images[domain]["boxes"],
                            images[domain][target_name]["boxes"])
            return loss
        return self.object_detection_loss(
            images[domain]["boxes"], images[domain]["fake_boxes"]
            ) if images[domain]["boxes"] is not None else 0

    def _cycle_object_detection_loss(self, images, domain):
        return self.object_detection_loss(
            images[domain]["boxes"],
            images[domain]["reconstruction_boxes"]) \
            if images[domain]["boxes"] is not None else 0

    def _generator_loss(self, images):
        generator_loss = super()._generator_loss(images)

        object_detection_loss = sum([
            self._object_detection_loss(images, domain)
            for domain in self.options.domain_names])
        if self.options.cycle_object_detection_loss_weight > 0:
            cycle_object_detection_loss = sum([
                self._cycle_object_detection_loss(images, domain)
                for domain in self.options.domain_names])
        else:
            cycle_object_detection_loss = 0

        generator_loss += (self.options.object_detection_loss_weight *
                           object_detection_loss) + \
            (self.options.cycle_object_detection_loss_weight *
             cycle_object_detection_loss)

        if self.options.verbose:
            self.iou_loss += (
                self.options.object_detection_loss_weight *
                object_detection_loss) + \
                (self.options.cycle_object_detection_loss_weight *
                 cycle_object_detection_loss)

        return generator_loss

    @classmethod
    def update_arguments(cls, options):
        """Update arguments"""
        super().update_arguments(options)
        options.parser.add_argument(
            '--learning-rate-detection', type=float, default=0.0002,
            dest="learning_rate_detection",
            help="Initial learning rate for detection net")
        #### Ground-truth handling ####
        options.parser.add_argument(
            '--always-use-pseudo-ground-truth', action='store_true',
            dest="always_use_pseudo_ground_truth",
            help="Always use a generated ground truth")
        options.parser.add_argument(
            '--use-pseudo-ground-truth-if-unavailable', action='store_true',
            dest="use_pseudo_ground_truth_if_unavailable",
            help="Use a generated ground truth if no ground truth is available")
        #### Loss weights ####
        options.parser.add_argument(
            '--object-detection-loss-weight', type=float, default=2,
            dest="object_detection_loss_weight",
            help="Weight for object detection loss")
        options.parser.add_argument(
            '--cycle-object-detection-loss-weight', type=float, default=1,
            dest="cycle_object_detection_loss_weight",
            help="Weight for cycle object detection loss")

        options.parser.add_argument(
            '--num-classes', type=int, default=91, dest="num_classes",
            help="Number of classes for object detection")

    def _evaluate_image(self, *args, **kwargs):
        outputs = super()._evaluate_image(*args, **kwargs)
        transfer = None
        save_path = None
        image_file_name = os.path.basename(
            kwargs["image_path"]).split('.')[0]
        if isinstance(outputs, tuple):
            # if several outputs: reconstructed + transfered images
            transfer = outputs[1]
            save_path = os.path.join(
                self.options.save_path,
                f"from_{self._direction_folder(args[1])}",
                f"{image_file_name}_epoch{kwargs['epoch_index']}.json"
                if kwargs['epoch_index'] is not None
                else f"{image_file_name}.json")
        else:
            # if one output: transfered image for multi-modal model
            transfer = outputs
            save_path = os.path.join(
                self.options.save_path,
                f'from_{args[1]}_to_{args[2]}',
                f'{image_file_name}_epoch{kwargs["epoch_index"]}.json'
                if kwargs["epoch_index"] is not None
                else f'{image_file_name}.json')
        bbox_prediction = self.detection_net(transfer)[0]
        # save bounding boxes
        with open(save_path, "w") as file_:
            json.dump(
                {key: value.tolist()
                 for key, value in bbox_prediction.items()},
                file_)
        return outputs
