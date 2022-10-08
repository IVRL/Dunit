"""ConcatGenerator for DRIT"""
from ..base_module import BaseModule
from .content_encoder import ContentEncoder
from .concat_style_encoder import ConcatStyleEncoder
from .concat_decoder import ConcatDecoder

class ConcatGenerator(BaseModule):
    """ConcatGenerator for DRIT"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.content_encoder = ContentEncoder(**kwargs)
        self.style_encoder = ConcatStyleEncoder(**kwargs)
        self.decoder = ConcatDecoder(**kwargs)

    def encode(self, image):
        """Encode images to content + style features"""
        return self.content_encoder(image), self.style_encoder(image)

    def decode(self, content, style):
        """Decode content + style features to images"""
        return self.decoder(content, style)
