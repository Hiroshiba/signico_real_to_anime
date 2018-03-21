import chainer

from deep_image_converter import utility

from .base_network import BaseNetwork


class DeepConv(BaseNetwork):
    def __init__(self, num_scale=2, base_num_z=16, **kwargs):
        super().__init__(
            conv1=utility.chainer.Convolution2D(None, base_num_z, 3, 1, 1),
            bn1=chainer.links.BatchNormalization(base_num_z),
            **kwargs,
        )
        self.num_scale = num_scale

        for i in range(num_scale):
            l = base_num_z * 2 ** (i + 1)
            self.add_link('conv{}'.format((i + 1) * 2), utility.chainer.Convolution2D(None, l, 4, 2, 1))
            self.add_link('conv{}'.format((i + 1) * 2 + 1), utility.chainer.Convolution2D(None, l, 3, 1, 1))
            self.add_link('bn{}'.format((i + 1) * 2), chainer.links.BatchNormalization(l))
            self.add_link('bn{}'.format((i + 1) * 2 + 1), chainer.links.BatchNormalization(l))

    def get_scaled_width(self, base_width):
        return base_width // (2 ** self.num_scale)

    def __call__(self, x, test):
        h = x
        num_layer = self.num_scale * 2 + 1
        for i in range(num_layer):
            conv = getattr(self, 'conv{}'.format(i + 1))
            bn = getattr(self, 'bn{}'.format(i + 1))
            h = super().activation(bn(conv(h), test=test), is_last=(i == num_layer - 1))
        return h


class PoorDeepConv(BaseNetwork):
    def __init__(self, num_scale=2, base_num_z=16, **kwargs):
        super().__init__(**kwargs)
        self.num_scale = num_scale

        for i in range(num_scale):
            l = base_num_z * 2 ** i
            self.add_link('conv{}'.format(i + 1), utility.chainer.Convolution2D(None, l, 4, 2, 1))
            self.add_link('bn{}'.format(i + 1), chainer.links.BatchNormalization(l))

    def get_scaled_width(self, base_width):
        return base_width // (2 ** self.num_scale)

    def __call__(self, x, test):
        h = x
        for i in range(self.num_scale):
            conv = getattr(self, 'conv{}'.format(i + 1))
            bn = getattr(self, 'bn{}'.format(i + 1))
            h = super().activation(bn(conv(h), test=test), is_last=(i == self.num_scale - 1))
        return h


class UnetConv(DeepConv):
    def __call__(self, x, test):
        h = x
        h_list = []
        num_layer = self.num_scale * 2 + 1
        for i in range(num_layer):
            conv = getattr(self, 'conv{}'.format(i + 1))
            bn = getattr(self, 'bn{}'.format(i + 1))
            h = super().activation(bn(conv(h), test=test), is_last=(i % 2 == 0))

            if i % 2 == 0:
                h_list.append(h)

        return list(reversed(h_list))
