from abc import ABCMeta, abstractmethod
import chainer
import typing

from deep_image_converter.config import ModelConfig


class BaseModel(chainer.Chain, metaclass=ABCMeta):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config


class BaseEncoder(chainer.Chain):
    def __init__(
            self,
            config: ModelConfig,
            encode_network,
            feature_network,
    ):
        super().__init__(
            encode=encode_network,
            feature=feature_network,
        )
        self.config = config

    def __call__(self, x, test):
        h = x
        h = h_encoded = self.encode(h, test)
        h = h_feature = self.feature(h, test)
        return h, {
            'encoded': h_encoded,
            'feature': h_feature,
        }

    def get_random_feature(self, batchsize):
        raise NotImplementedError()


class BaseDecoder(BaseModel):
    def __init__(
            self,
            config: ModelConfig,
            decode_network,
            coloring_network,
            pre_decode_network=None,
    ):
        super().__init__(
            config=config,
            decode=decode_network,
            coloring=coloring_network,
        )

        if pre_decode_network is not None:
            self.add_link('pre_decode_network', pre_decode_network)

    def __call__(self, x, test):
        h = x

        if hasattr(self, 'pre_decode_network'):
            h = self.pre_decode_network(h, test)

        h = h_decoded = self.decode(h, test)
        h = h_coloring = self.coloring(h, test)
        return h, {
            'decoded': h_decoded,
            'image': h_coloring,
        }


class BaseAutoEncoderModel(BaseModel, metaclass=ABCMeta):
    def __call__(self, x, test) -> (chainer.Variable, typing.Dict):
        h = x
        h, h_encoded = self.encode(h, test)
        h, h_decoded = self.decode(h, test)

        image = h

        return image, {
            'encoded': h_encoded,
            'decoded': h_decoded,
        }

    def encode(self, x, test):
        return self.enc(x, test)

    def decode(self, x, test):
        return self.dec(x, test)

    def get_random_feature(self, batchsize):
        return self.enc.get_random_feature(batchsize)


class BaseConvertModel(BaseModel, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, x, test) -> (chainer.Variable, typing.Dict):
        pass

    @abstractmethod
    def encode(self, x, test) -> (chainer.Variable, typing.Dict):
        pass

    @abstractmethod
    def decode(self, x, test) -> (chainer.Variable, typing.Dict):
        pass


class BaseEachConvertModel(BaseModel, metaclass=ABCMeta):
    def __call__(self, x, from_a_b: str, to_a_b: str, test):
        h = x
        h, h_encoded = self.encode(h, from_a_b, test)
        h, h_decoded = self.decode(h, to_a_b, test)

        image = h

        return image, {
            'encoded': h_encoded,
            'decoded': h_decoded,
        }

    def encode(self, x, from_a_b: str, test):
        assert from_a_b == 'a' or from_a_b == 'b'

        h = x
        h = h_encoded = self.enc(h, test)
        h = h_feature = self.feature(h, test)
        return h, {
            'encoded': h_encoded,
            'feature': h_feature,
        }

    def decode(self, x, to_a_b: str, test):
        assert to_a_b == 'a' or to_a_b == 'b'

        if to_a_b == 'a':
            dec = self.a_dec
            coloring = self.a_coloring
        else:
            dec = self.b_dec
            coloring = self.b_coloring

        h = x
        h = h_decoded = dec(h, test)
        h = h_coloring = coloring(h, test)
        return h, {
            'decoded': h_decoded,
            'image': h_coloring,
        }
