"""Mixin for evaluating multi-directional models"""
import os
from torchvision.utils import save_image

class MultiDirectionalEvaluationMixin():#pylint: disable=too-few-public-methods
    """Mixin for evaluating multi-directional models"""
    multimodal = True

    def evaluate(self, epoch_index, *image_batches):
        """Save an example of images transfered by the model"""
        if not isinstance(image_batches[0][0], tuple):
            image_batches = [
                (("source",), *image_batches[0]),
                (("target",), *image_batches[1])]

        for source_domain, images, file_paths in image_batches:
            source_domain = source_domain[0]
            images = images["image"]
            if source_domain in (
                    getattr(self.options, "source_domains",
                            self.options.domain_names)):
                for image, file_path in zip(images, file_paths):
                    for target_domain in (
                            getattr(self.options, "target_domains",
                                    self.options.domain_names)):
                        transfer = self._evaluate_image(
                            image, source_domain, target_domain,
                            image_path=file_path, epoch_index=epoch_index)

                        file_name = os.path.splitext(
                            os.path.basename(file_path))[0]
                        # save images
                        save_image(
                            [image, transfer[0]],
                            os.path.join(
                                self.options.save_path,
                                f'from_{source_domain}_to_{target_domain}',
                                f'{file_name}_epoch{epoch_index}.png'
                                if epoch_index is not None
                                else f'{file_name}.png'),
                            nrow=2)

    def _folder_creation(self):
        """Create folder architecture for saving images"""
        for source_domain in getattr(self.options, "source_domains",
                                     self.options.domain_names):
            for target_domain in getattr(self.options, "target_domains",
                                         self.options.domain_names):
                folder_path = os.path.join(
                    self.options.save_path,
                    f'from_{source_domain}_to_{target_domain}')
                if not os.path.isdir(folder_path):
                    os.makedirs(folder_path)

    def _evaluate_image(self, image, source_domain, target_domain, image_path,
                        *args, epoch_index=None, **kwargs):
        """Generate image for model"""
        raise NotImplementedError("Image evaluation is not defined")
