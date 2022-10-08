"""Model for object segmentation based on DRIT"""
import os
import pickle
import time
import torch

from ..drit.model import DRIT
from .mixin import ObjectDetectionMixin

class DRITObjectDetection(
        ObjectDetectionMixin, DRIT):
    """Model for object segmentation based on DRIT"""
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
                    if len(domain_data) == 3:
                        image, image_paths, bboxes = domain_data
                    else:
                        _, image, image_paths, bboxes = domain_data
                    image = image.to(self.device)
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
