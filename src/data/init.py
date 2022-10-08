"""Dataset for INIT"""
import torch
from .image_folder_with_annotations import ImageFolderWithAnnotationsDataset

class INITDataset(ImageFolderWithAnnotationsDataset):#pylint: disable=too-few-public-methods
    """Dataset for INIT"""
    def __init__(self, *args, **kwargs):
        args[0].recursive = True
        super().__init__(*args, **kwargs)
        self.bboxes = {}
        with open(self.annotation_path, 'r') as file_:
            while file_.readline():
                # first line is the index
                # second is the path to the image
                path = file_.readline()[:-1]
                self.bboxes[path] = []
                # third is the dimensions of the image
                file_.readline()
                # fourth is 0
                file_.readline()
                # fifth is the number of bounding boxes
                nb_bounding_boxes = int(file_.readline()[:-1])
                # subsequent lines are `label x1 y1 x2 y2` for the bboxes
                for _ in range(nb_bounding_boxes):
                    infos = file_.readline()[:-1].split()
                    self.bboxes[path].append({
                        "label": int(infos[0]),
                        "bbox": [float(coord) for coord in infos[1:]]})
                # end with 2 empty lines
                file_.readline()
                file_.readline()

        self.bboxes = {
            key: {
                "boxes": self.to_local_tensor([bbox["bbox"] for bbox in value]),
                "labels": self.to_local_tensor(
                    [bbox["label"] for bbox in value], dtype=torch.int64)
                }
            for key, value in self.bboxes.items()}


    def _get_annotation(self, index, image_path):# pylint: disable=unused-argument
        return self.bboxes[image_path]
