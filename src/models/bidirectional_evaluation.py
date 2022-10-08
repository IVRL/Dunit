"""Mixin for evaluating bi-directional models"""
import os
from torchvision.utils import save_image

class BidirectionalEvaluationMixin():#pylint: disable=too-few-public-methods
    """Mixin for evaluating bi-directional models"""
    use_lists = None
    multimodal = False

    def evaluate(self, epoch_index, source_images, target_images):
        """Save an example of images transfered by the model"""
        directions = self._get_directions()
        for direction in directions:
            for image, file_path, *_others in zip(
                    *(source_images if direction else target_images)):
                reconstruction, transfer = self._evaluate_image(
                    image, direction)

                file_name = os.path.splitext(os.path.basename(file_path))[0]
                # save images
                for joint_image, suffix in [
                        (reconstruction[0], "reconstruction"),
                        (transfer[0], "transfer")]:
                    save_image(
                        [image, joint_image],
                        os.path.join(
                            self.options.save_path,
                            f"from_{self._direction_folder(direction)}",
                            f"{file_name}_epoch{epoch_index}_{suffix}.png"),
                        nrow=2)

    def _get_directions(self):
        """Retrieve needed evaluation directions"""
        if self.use_lists is None:
            self.use_lists = isinstance(self.generators, list)
        if self.use_lists:
            if getattr(self.options, 'direction', 'both') == 'both':
                return [0, 1]
            if self.options.direction == 'to_source':
                return [0]
            return [1]
        if getattr(self.options, 'direction', 'both') == 'both':
            return ["source", "target"]
        return [self.options.direction[3:]]

    def _negate_direction(self, direction):
        """Negation of the direction"""
        if self.use_lists:
            return not direction
        return "source" if direction == "target" else "target"

    def _direction_folder(self, direction):
        """Retrieve the folder in which to save the images"""
        if self.use_lists:
            return "source" if direction else "target"
        return direction

    def _evaluate_image(self, image, direction):
        """Generate reconstructed image and transfered image for model"""
        raise NotImplementedError("Image evaluation is not defined")
