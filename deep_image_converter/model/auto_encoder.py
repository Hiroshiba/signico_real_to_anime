from .base_model import BaseAutoEncoderModel
from .encoder import *
from .decoder import *


class DeepAutoEncoder(BaseAutoEncoderModel):
    def __init__(self, config: ModelConfig):
        nume = config.num_encoder_block
        numd = config.num_decoder_block

        decoder_class = choose_coloring_decoder(config)
        super().__init__(
            config,
            enc=ConvFeatureEncoder(
                config=config,
                encode_network=network.DeepConv(
                    num_scale=nume,
                    base_num_z=config.num_z_base_encoder,
                ),
            ),

            dec=decoder_class(
                config=config,
                pre_decode_network=network.DeconvVector(
                    config.num_z_base_decoder * 2 ** numd,
                    config.image_width // 2 ** numd,
                ),
                decode_network=network.DeepDeconv(
                    num_scale=numd,
                    base_num_z=config.num_z_base_decoder,
                    last_activation=chainer.functions.sigmoid if decoder_class == MuVarColoringDecoder else None,
                ),
            ),
        )


class DeepVariationalAutoEncoder(BaseAutoEncoderModel):
    def __init__(self, config: ModelConfig):
        nume = config.num_encoder_block
        numd = config.num_decoder_block

        super().__init__(
            config,
            enc=MuVarFeatureEncoder(
                config=config,
                encode_network=network.DeepConv(
                    num_scale=nume,
                    base_num_z=config.num_z_base_encoder,
                    last_activation=chainer.functions.sigmoid,
                ),
            ),
        )

        decoder_class = choose_coloring_decoder(config)
        self.add_link('dec', decoder_class(
            config=config,
            pre_decode_network=network.DeconvVector(
                config.num_z_base_decoder * 2 ** numd,
                config.image_width // 2 ** numd,
            ),
            decode_network=network.DeepDeconv(
                num_scale=numd,
                base_num_z=config.num_z_base_decoder,
                last_activation=chainer.functions.sigmoid if decoder_class == MuVarColoringDecoder else None,
            ),
        ))


class UnetVariationalAutoEncoder(BaseAutoEncoderModel):
    def __init__(self, config: ModelConfig):
        if 'unet_ksize_feature_list' in config.other:
            size_list = [config.image_width // 2 ** 4] + config.other['unet_ksize_feature_list']
        else:
            size_list = [config.image_width // 2 ** 4, 2, 2, 2, 2]

        if 'unet_num_z_feature_list' in config.other:
            num_z_list = [config.num_z_feature] + config.other['unet_num_z_feature_list']
        else:
            num_z_list = [config.num_z_feature]
            for size in size_list[:-1]:
                num_z_list += [num_z_list[-1] // size]

        for num_z, size in zip(num_z_list, size_list):
            assert (num_z is None) == (size is None)

        super().__init__(
            config,
            enc=MultiMuVarFeatureEncoder(
                config=config,
                encode_network=network.UnetConv(
                    num_scale=4,
                    base_num_z=config.num_z_base_encoder,
                    last_activation=chainer.functions.sigmoid,
                ),
                num_z_list=num_z_list,
                size_list=size_list,
            ),
        )

        out_channels_list = [config.num_z_base_encoder * 2 ** i for i in reversed(range(4 + 1))]

        decoder_class = choose_coloring_decoder(config)
        self.add_link('dec', decoder_class(
            config=config,
            pre_decode_network=network.MultiDeconvVector(
                out_channels_list=out_channels_list,
                ksize_list=size_list,
            ),
            decode_network=network.UnetDeconv(
                num_scale=4,
                base_num_z=config.num_z_base_decoder,
            ),
        ))


class ConcatVariationalAutoEncoder(BaseAutoEncoderModel):
    def __init__(self, config: ModelConfig):
        split = numpy.cumsum(config.other['num_z_feature_list'])
        assert split[-1] == config.num_z_feature
        split = split[:-1]

        super().__init__(
            config,
            enc=MuVarFeatureEncoder(
                config=config,
                encode_network=network.DeepConv(
                    num_scale=4,
                    base_num_z=config.num_z_base_encoder,
                    last_activation=chainer.functions.sigmoid,
                ),
            ),
        )

        decoder_class = choose_coloring_decoder(config)
        self.add_link('dec', decoder_class(
            config=config,
            pre_decode_network=utility.chainer.ChainList(
                utility.chainer.FunctionLink(
                    function=chainer.functions.SplitAxis(split, axis=1)
                ),
                network.FirstDeconvVector(
                    config.num_z_base_decoder * 2 ** 4,
                    config.image_width // 2 ** 4,
                ),
            ),
            decode_network=network.ConcatDeconv(
                num_scale=4,
                base_num_z=config.num_z_base_decoder,
            ),
        ))
