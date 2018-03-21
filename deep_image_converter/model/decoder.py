import chainer
import numpy

from deep_image_converter import network
from deep_image_converter.config import ModelConfig
from deep_image_converter import utility

from .base_model import BaseDecoder


def choose_coloring_decoder(config: ModelConfig):
    if config.method_output['name'] == 'normal_distribution':
        decoder_class = MuVarColoringDecoder
    elif config.method_output['name'] == 'binary_distribution':
        decoder_class = BinaryColoringDecoder
    elif config.method_output['name'] == 'regression':
        decoder_class = ConvColoringDecoder
    elif config.method_output['name'] == 'softmax_distribution':
        decoder_class = SoftmaxColoringDecoder
    else:
        raise ValueError(config.method_output['name'])
    return decoder_class


class _OneConv(chainer.Chain):
    def __init__(self, out_channels: int, ksize: int):
        super().__init__(
            conv=chainer.links.Convolution2D(None, out_channels, ksize, 1, 0)
        )

    def __call__(self, x, test):
        return self.conv(x)


class ConvColoringDecoder(BaseDecoder):
    def __init__(
            self,
            config: ModelConfig,
            decode_network,
            pre_decode_network=None,
    ):
        super().__init__(
            config=config,
            decode_network=decode_network,
            coloring_network=network.ConvColoring(channel=config.method_output['channel']),
            pre_decode_network=pre_decode_network,
        )


class MuVarColoringDecoder(BaseDecoder):
    def __init__(
            self,
            config: ModelConfig,
            decode_network,
            pre_decode_network=None,
    ):
        super().__init__(
            config=config,
            decode_network=decode_network,
            coloring_network=network.VaeMuVarVertor(num_z=config.method_output['channel'], size=1),
            pre_decode_network=pre_decode_network,
        )

    def __call__(self, x, test):
        h = x
        h, other = super().__call__(h, test)
        other['mu'], other['var'] = h

        sampling = lambda _h: chainer.functions.gaussian(_h[0], _h[1]) if not test else _h[0]

        h = sampling(h)
        other['image'] = h
        other['sampling'] = sampling
        return h, other


class BinaryColoringDecoder(BaseDecoder):
    def __init__(
            self,
            config: ModelConfig,
            decode_network,
            pre_decode_network=None,
    ):
        super().__init__(
            config=config,
            decode_network=decode_network,
            coloring_network=_OneConv(out_channels=config.method_output['channel'], ksize=1),
            pre_decode_network=pre_decode_network,
        )

    def __call__(self, x, test):
        h = x
        h, other = super().__call__(h, test)
        other['b'] = h

        h = chainer.functions.tanh(h)
        other['image'] = h
        return h, other


class SoftmaxColoringDecoder(BaseDecoder):
    def __init__(
            self,
            config: ModelConfig,
            decode_network,
            pre_decode_network=None,
    ):
        super().__init__(
            config=config,
            decode_network=decode_network,
            coloring_network=_OneConv(config.method_output['num_bin'] * config.method_output['channel'], 1),
            pre_decode_network=pre_decode_network,
        )

    def __call__(self, x, test):
        h = x
        h, other = super().__call__(h, test)

        num_bin = self.config.method_output['num_bin']
        h = chainer.functions.split_axis(h, (num_bin * 1, num_bin * 2), axis=1)
        h = chainer.functions.stack(h, axis=2)

        image = chainer.functions.argmax(h, axis=1)
        image = chainer.Variable(image.data.astype(numpy.float32) / (num_bin / 2) - 1)

        other['image'] = h
        return image, other
