import chainer

from deep_image_converter import utility


class BaseResidualBlock(chainer.Chain):
    def _padding_channel(self, h, x, test):
        if x.data.shape != h.data.shape:
            n, c, hh, ww = x.data.shape
            pad_c = h.data.shape[1] - c
            p = self.xp.zeros((n, pad_c, hh, ww), dtype=self.xp.float32)
            p = chainer.Variable(p, volatile='auto')
            x = chainer.functions.concat((p, x))

        return x


class ResidualBlock(BaseResidualBlock):
    def __init__(self, num_layer):
        super().__init__(
            conv1=utility.chainer.Convolution2D(None, num_layer, ksize=3, stride=1, pad=1),
            bn1=chainer.links.BatchNormalization(num_layer),
            conv2=utility.chainer.Convolution2D(num_layer, num_layer, ksize=3, stride=1, pad=1),
            bn2=chainer.links.BatchNormalization(num_layer),
        )

    def __call__(self, x, test):
        h = x
        h = chainer.functions.leaky_relu(self.bn1(self.conv1(h), test=test))
        h = self.bn2(self.conv2(h), test=test)
        x = self._padding_channel(h, x, test)
        return x + h


class DilateResidualBlock(BaseResidualBlock):
    def __init__(self, num_layer, dilate):
        super().__init__(
            dilate=utility.chainer.DilatedConvolution2D(None, num_layer, 3, 1, dilate, dilate=dilate),
            bn1=chainer.links.BatchNormalization(num_layer),
            conv=utility.chainer.Convolution2D(num_layer, num_layer, 1, 1, 0),
            bn2=chainer.links.BatchNormalization(num_layer),
        )

    def __call__(self, x, test):
        h = x
        h = chainer.functions.leaky_relu(self.bn1(self.dilate(h), test=test))
        h = self.bn2(self.conv(h), test=test)

        x = self._padding_channel(h, x, test)
        return x + h
