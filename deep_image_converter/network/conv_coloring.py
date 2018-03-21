import chainer

from deep_image_converter import utility


class ConvColoring(chainer.Chain):
    def __init__(self, channel=3):
        super().__init__(
            conv=chainer.links.Convolution2D(None, channel, 1),
        )

    def __call__(self, x, test):
        h = x
        h = chainer.functions.tanh(self.conv(h))
        return h
