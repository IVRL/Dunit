"""Generator for DRIT"""
from ..base_module import BaseModule
from .content_encoder import ContentEncoder
from .style_encoder import StyleEncoder
from .decoder import Decoder

class Generator(BaseModule):
    """Generator for DRIT"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.content_encoder = ContentEncoder(**kwargs)
        self.style_encoder = StyleEncoder(**kwargs)
        self.decoder = Decoder(**kwargs)

    def encode(self, image, domain):
        """Encode images to content + style features"""
        return self.encode_content(image), self.encode_style(image, domain)

    def encode_content(self, image):
        """Encode image to content features"""
        return self.content_encoder(image)

    def encode_style(self, image, domain):
        """Encode image to style features"""
        return self.style_encoder(image, domain)

    def decode(self, content, style, domain):
        """Decode content + style features to images"""
        return self.decoder(content, style, domain)
