import chainer

from deep_image_converter.config import ModelConfig

from .base_model import BaseModel
from .co_encode_model import CoEncodeModel
from .co_encode_model import OneDimensionLatentCoEncodeModel
from .convert_model import PoorDilateConvertModel
from .auto_encoder import DeepAutoEncoder
from .auto_encoder import DeepVariationalAutoEncoder
from .auto_encoder import UnetVariationalAutoEncoder
from .auto_encoder import ConcatVariationalAutoEncoder
from .discriminator import BaseDiscriminator
from .discriminator import Discriminator
from .discriminator import PoorDiscriminator


def prepare_model(model_config: ModelConfig, skip_load_model=False) -> BaseModel:
    model = None
    if model_config.name == 'co_encode_model':
        model = CoEncodeModel(model_config)
    elif model_config.name == 'one_dimension_latent_co_encode_model':
        model = OneDimensionLatentCoEncodeModel(model_config)
    elif model_config.name == 'poor_dilate_convert_model':
        model = PoorDilateConvertModel(model_config)
    elif model_config.name == 'deep_ae_model':
        model = DeepAutoEncoder(model_config)
    elif model_config.name == 'deep_vae_model':
        model = DeepVariationalAutoEncoder(model_config)
    elif model_config.name == 'unet_vae_model':
        model = UnetVariationalAutoEncoder(model_config)
    elif model_config.name == 'deep_concat_vae_model':
        model = ConcatVariationalAutoEncoder(model_config)
    else:
        assert "{name} is not defined.".format(name=model_config.name)

    if not skip_load_model and model_config.pre_trained_path is not None:
        chainer.serializers.load_npz(model_config.pre_trained_path, model)

    return model


def choose_discriminator(model_config: ModelConfig) -> BaseDiscriminator:
    dis = None
    if model_config.discriminator == 'discriminator':
        dis = Discriminator(model_config)
    elif model_config.discriminator == 'poor_discriminator':
        dis = PoorDiscriminator(model_config)
    else:
        assert "{name} is not defined.".format(name=model_config.name)
    return dis
