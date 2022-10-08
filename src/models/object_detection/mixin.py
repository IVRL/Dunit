"""Mixin for object detection"""
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ..losses import IoULoss, multibox_generalized_intersection_over_union, \
    multibox_intersection_over_union
from ..utils import attempt_use_apex

USE_APEX, AMP = attempt_use_apex()

class ObjectDetectionMixin():
    """Mixin for object detection"""
    def _end_setup(self, use_lists=False):
        self.object_detection_loss = IoULoss(
            iou_func=(
                multibox_generalized_intersection_over_union
                if self.options.iou_func == 'generalized'
                else multibox_intersection_over_union),
            reduction=self.options.loss_reduction,
            area_normalization=self.options.area_normalization)
        self.detection_net = fasterrcnn_resnet50_fpn(
            pretrained=False, pretrained_backbone=False,
            num_classes=self.options.num_classes).to(self.device)
        self.iou_loss = 0

        self._create_optimizer([self.detection_net], True, "detection")
        super()._end_setup(use_lists)

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
        *source_images, _, images["source"]["segmentation"] = source_images
        *target_images, _, images["target"]["segmentation"] = target_images

        super()._generate_training_images(source_images, target_images, images)

        for domain in ["source", "target"]:
            if images[domain]["segmentation"] is not None:
                boxes = []
                for index in range(images[domain]['real'].size(0)):
                    boxes.append({
                        "boxes": images[domain]["segmentation"]["boxes"][index],
                        "labels": images[domain]["segmentation"]["labels"][
                            index]})
                images[domain]["segmentation"] = boxes
            # if wanted/needed, get pseudo ground truth
            if self.options.always_use_pseudo_ground_truth or (
                    self.options.use_pseudo_ground_truth_if_unavailable and
                    images[domain]["segmentation"] is None):
                images[domain]["segmentation"] = self.detection_net(
                    images[domain]["real"])

            # compute object detection for generated images
            if images[domain]["segmentation"] is not None:
                images[domain]["fake_segmentation"] = self.detection_net(
                    images[self._negate_direction(domain)]["fake"])

                # compute object detection for reconstructed images
                if self.options.cycle_object_detection_loss_weight > 0:
                    images[domain]["reconstruction_segmentation"] = \
                        self.detection_net(images[domain]["reconstruction"])

    def _generate_training_data(self, input_images, data):
        """Used for multi-modal models"""
        input_images = list(input_images)
        for index, (domain, *images, bboxes) in enumerate(input_images):
            data[domain[0]]["segmentation"] = bboxes
            input_images[index] = (domain, *images)

        super()._generate_training_data(input_images, data)

        for domain, images, _ in input_images:
            domain = domain[0]
            if data[domain]['segmentation']:
                boxes = []
                for index in range(data[domain]['real'].size(0)):
                    boxes.append({
                        "boxes": data[domain]["segmentation"]["boxes"][index],
                        "labels": data[domain]["segmentation"]["labels"][index]
                        })
                data[domain]["segmentation"] = boxes
            # if wanted/needed, get pseudo ground truth
            if self.options.always_use_pseudo_ground_truth or (
                    self.options.use_pseudo_ground_truth_if_unavailable and
                    data[domain]["segmentation"] is None):
                data[domain]["segmentation"] = self.detection_net(
                    data[domain]["real"])

            if data[domain]["segmentation"] is not None:
                for target_domain in self.options.domain_names:
                    if target_domain != domain:
                        # compute object detection for generated images
                        data[domain][target_domain]["segmentation"] = \
                            self.detection_net(
                                data[domain][target_domain]["fake"])

                # compute object detection for reconstructed images
                if self.options.cycle_object_detection_loss_weight > 0:
                    data[domain]["reconstruction_segmentation"] = \
                        self.detection_net(data[domain]["reconstruction"])

    def _object_detection_loss(self, images, domain):
        if self.multimodal:
            loss = 0
            if images[domain]["segmentation"] is not None:
                for target_name in self.options.domain_names:
                    if target_name != domain:
                        loss += self.object_detection_loss(
                            images[domain]["segmentation"],
                            images[domain][target_name]["segmentation"])
            return loss
        return self.object_detection_loss(
            images[domain]["segmentation"], images[domain]["fake_segmentation"]
            ) if images[domain]["segmentation"] is not None else 0

    def _cycle_object_detection_loss(self, images, domain):
        return self.object_detection_loss(
            images[domain]["segmentation"],
            images[domain]["reconstruction_segmentation"]) \
            if images[domain]["segmentation"] is not None else 0

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
                object_detection_loss.item()) + \
                (self.options.cycle_object_detection_loss_weight *
                 cycle_object_detection_loss.item())

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
        #### IoU loss parameters ####
        options.parser.add_argument(
            '--iou-func', type=str, choices=["generalized", "simple"],
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
