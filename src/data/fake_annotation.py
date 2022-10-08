"""Dataset faking annotations"""
from .image_folder_with_annotations import ImageFolderWithAnnotationsDataset

class FakeAnnotationDataset(ImageFolderWithAnnotationsDataset):#pylint: disable=too-few-public-methods
    """Dataset faking annotations"""
    def _get_annotation(self, index, image_path):# pylint: disable=unused-argument
        return []
