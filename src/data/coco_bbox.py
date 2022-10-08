"""Dataset for Coco Object detection"""
import json
import os

from .image_folder_with_annotations import ImageFolderWithAnnotationsDataset

class CocoBboxDataset(ImageFolderWithAnnotationsDataset):#pylint: disable=too-few-public-methods
    """Dataset for Coco Object detection"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(self.annotation_path, 'r') as file_:
            dataset = json.load(file_)

        image_ids = {image["id"]: image["file_name"]
                     for image in dataset["images"]}
        self.bboxes = {}
        for annotation in dataset["annotations"]:
            image_file_name = image_ids[annotation["image_id"]]
            if image_file_name not in self.bboxes.keys():
                self.bboxes[image_file_name] = [{
                    "bbox": [
                        annotation["bbox"][0],
                        annotation["bbox"][1],
                        annotation["bbox"][0] + annotation["bbox"][2],
                        annotation["bbox"][1] + annotation["bbox"][3],
                        ],
                    "category": annotation["category_id"]}]
            else:
                self.bboxes[image_file_name].append({
                    "bbox": [
                        annotation["bbox"][0],
                        annotation["bbox"][1],
                        annotation["bbox"][0] + annotation["bbox"][2],
                        annotation["bbox"][1] + annotation["bbox"][3],
                        ],
                    "category": annotation["category_id"]})
        self.bboxes = {
            key: {
                "boxes": [bbox["bbox"] for bbox in value],
                "labels": [bbox["category"] for bbox in value]
                }
            for key, value in self.bboxes.items()}

    def _get_annotation(self, index, image_path):# pylint: disable=unused-argument
        return self.bboxes[os.path.basename(image_path)]
