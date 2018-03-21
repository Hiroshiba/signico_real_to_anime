import chainer

from deep_image_converter import network
from deep_image_converter.config import ModelConfig

from .base_model import BaseConvertModel


class PoorDilateConvertModel(BaseConvertModel):
    def __init__(self, config: ModelConfig):
        super().__init__(
            config,
            enc=network.PoorDilateEncoder(
                num_z_residual=config.num_z_base_encoder,
                num_residual=config.num_encoder_block,
            ),

            dec=network.PoorDilateDecoder(
                num_z_residual=config.num_z_base_decoder,
                num_residual=config.num_decoder_block,
            ),
            coloring=network.ConvColoring(),
        )

    def __call__(self, x, test):
        h = x
        h, h_encoded = self.encode(h, test)
        h, h_decoded = self.decode(h, test)

        image = h

        return image, {
            'encoded': h_encoded,
            'decoded': h_decoded,
        }

    def encode(self, x, test):
        h = x
        h = h_encoded = self.enc(h, test)
        return h, {
            'encoded': h_encoded,
        }

    def decode(self, x, test):
        h = x
        h = h_decoded = self.dec(h, test)
        h = h_coloring = self.coloring(h, test)
        return h, {
            'decoded': h_decoded,
            'image': h_coloring,
        }
