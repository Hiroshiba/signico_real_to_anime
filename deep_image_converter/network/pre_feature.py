import chainer

from deep_image_converter import utility


class ConvFeature(chainer.Chain):
    def __init__(self, num_z):
        super().__init__(
            conv=chainer.links.Convolution2D(None, num_z, 1, 1, 0),
        )

    def __call__(self, x, test):
        h = x
        h = self.conv(h)
        return h


class ConvVector(chainer.Chain):
    def __init__(self, num_z, size):
        super().__init__(
            conv=chainer.links.Convolution2D(None, num_z, size, 1, 0),
        )

    def __call__(self, x, test):
        h = x
        h = chainer.functions.tanh(self.conv(h))
        return h


class VaeMuVarVertor(chainer.Chain):
    def __init__(self, num_z, size):
        super().__init__(
            conv_mu=chainer.links.Convolution2D(None, num_z, size, size, 0),
            conv_var=chainer.links.Convolution2D(None, num_z, size, size, 0),
        )

    def __call__(self, x, test):
        h = x
        h_mu = self.conv_mu(h)
        h_var = self.conv_var(h)
        return h_mu, h_var


class MultiVaeMuVarVertor(chainer.Chain):
    def __init__(self, num_z_list, size_list):
        super().__init__()
        for i, (num_z, size) in enumerate(zip(num_z_list, size_list)):
            if num_z is not None:
                self.add_link('mu_var{}'.format(i + 1), VaeMuVarVertor(num_z, size))

        self.num = len(num_z_list)

    def __call__(self, x, test):
        assert isinstance(x, list) and len(x) == self.num

        h_list = []
        for i, h in enumerate(x):
            key = 'mu_var{}'.format(i + 1)
            if not hasattr(self, key):
                h_list.append(None)
                continue

            mu_var = getattr(self, key)
            h_list.append(mu_var(h, test))
        return h_list
