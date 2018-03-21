import chainer

from deep_image_converter import network
from deep_image_converter.config import ModelConfig
from deep_image_converter import utility

from .base_model import BaseEachConvertModel


class CoEncodeModel(BaseEachConvertModel):
    def __init__(self, config: ModelConfig):
        super().__init__(
            config,
            enc=network.Encoder(
                num_z_residual=config.num_z_base_encoder,
                num_residual=config.num_encoder_block,
            ),
            feature=network.ConvFeature(num_z=config.num_z_feature),

            a_dec=network.Decoder(
                num_z_residual=config.num_z_base_decoder,
                num_residual=config.num_decoder_block,
            ),
            a_coloring=network.ConvColoring(channel=config.method_output['channel']),
            b_dec=network.Decoder(
                num_z_residual=config.num_z_base_decoder,
                num_residual=config.num_decoder_block,
            ),
            b_coloring=network.ConvColoring(channel=config.method_output['channel']),
        )


class OneDimensionLatentCoEncodeModel(BaseEachConvertModel):
    def __init__(self, config: ModelConfig):
        nume = config.num_encoder_block
        numd = config.num_decoder_block
        super().__init__(
            config,
            enc=network.DeepConv(
                num_scale=nume,
                base_num_z=config.num_z_base_encoder,
            ),
            feature=network.ConvVector(num_z=config.num_z_feature, size=config.image_width // 2 ** nume),

            a_dec=utility.chainer.ChainList(
                network.DeconvVector(
                    config.num_z_base_decoder * 2 ** numd,
                    config.image_width // 2 ** numd,
                ),
                network.PoorDeepDeconv(
                    num_scale=numd,
                    base_num_z=config.num_z_base_decoder,
                ),
                forwarder=[
                    lambda h, i, child, test: child(h, test=test),
                    lambda h, i, child, test: child(h, test=test),
                ],
            ),
            a_coloring=network.ConvColoring(channel=config.method_output['channel']),
            b_dec=utility.chainer.ChainList(
                network.DeconvVector(
                    config.num_z_base_decoder * 2 ** numd,
                    config.image_width // 2 ** numd,
                ),
                network.PoorDeepDeconv(
                    num_scale=numd,
                    base_num_z=config.num_z_base_decoder,
                ),
                forwarder=[
                    lambda h, i, child, test: child(h, test=test),
                    lambda h, i, child, test: child(h, test=test),
                ],
            ),
            b_coloring=network.ConvColoring(channel=config.method_output['channel']),
        )
