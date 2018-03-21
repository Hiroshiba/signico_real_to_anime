import chainer

from deep_image_converter import utility

from .base_network import BaseNetwork


class DeepDeconv(BaseNetwork):
    def __init__(self, num_scale=2, base_num_z=32, **kwargs):
        super().__init__(**kwargs)
        self.num_scale = num_scale

        self.add_link('deconv1', utility.chainer.Convolution2D(None, base_num_z * 2 ** num_scale, 3, 1, 1))
        self.add_link('bn1', chainer.links.BatchNormalization(base_num_z * 2 ** num_scale))

        for i in range(num_scale):
            l = base_num_z * 2 ** (num_scale - 1 - i)
            self.add_link('deconv{}'.format((i + 1) * 2), utility.chainer.Deconvolution2D(None, l, 4, 2, 1))
            self.add_link('deconv{}'.format((i + 1) * 2 + 1), utility.chainer.Convolution2D(None, l, 3, 1, 1))
            self.add_link('bn{}'.format((i + 1) * 2), chainer.links.BatchNormalization(l))
            self.add_link('bn{}'.format((i + 1) * 2 + 1), chainer.links.BatchNormalization(l))

    def get_scaled_width(self, base_width):
        return base_width // (2 ** self.num_scale)

    def __call__(self, x, test):
        h = x
        num_layer = self.num_scale * 2 + 1
        for i in range(num_layer):
            deconv = getattr(self, 'deconv{}'.format(i + 1))
            bn = getattr(self, 'bn{}'.format(i + 1))
            h = super().activation(bn(deconv(h), test=test), is_last=(i == num_layer - 1))
        return h


class PoorDeepDeconv(BaseNetwork):
    def __init__(self, num_scale=2, base_num_z=32, **kwargs):
        super().__init__(**kwargs)
        self.num_scale = num_scale

        for i in range(num_scale):
            l = base_num_z * 2 ** (num_scale - 1 - i)
            self.add_link('deconv{}'.format(i + 1), utility.chainer.Deconvolution2D(None, l, 4, 2, 1))
            self.add_link('bn{}'.format(i + 1), chainer.links.BatchNormalization(l))

    def get_scaled_width(self, base_width):
        return base_width // (2 ** self.num_scale)

    def __call__(self, x, test):
        h = x
        for i in range(self.num_scale):
            deconv = getattr(self, 'deconv{}'.format(i + 1))
            bn = getattr(self, 'bn{}'.format(i + 1))
            h = super().activation(bn(deconv(h), test=test), is_last=(i == self.num_scale - 1))
        return h


class UnetDeconv(DeepDeconv):
    def __call__(self, x, test):
        h_list = x
        assert isinstance(h_list, list)

        h = h_list.pop(0)
        for i in range(self.num_scale * 2 + 1):
            deconv = getattr(self, 'deconv{}'.format(i + 1))
            bn = getattr(self, 'bn{}'.format(i + 1))
            h = chainer.functions.leaky_relu(bn(deconv(h), test=test))

            if i % 2 == 1:
                h_u = h_list.pop(0)
                if h_u is not None:
                    h = chainer.functions.concat((h, h_u))
        return h


class ConcatDeconv(DeepDeconv):
    def __call__(self, x, test):
        h_list = x
        assert isinstance(h_list, list)

        h = h_list.pop(0)
        for i in range(self.num_scale * 2 + 1):
            deconv = getattr(self, 'deconv{}'.format(i + 1))
            bn = getattr(self, 'bn{}'.format(i + 1))
            h = chainer.functions.leaky_relu(bn(deconv(h), test=test))

            if i < self.num_scale * 2 and i % 2 == 0:
                h_u = h_list.pop(0)
                h_u = chainer.functions.broadcast_to(h_u, h_u.shape[:2] + h.shape[-2:])
                h = chainer.functions.concat((h, h_u))
        return h
