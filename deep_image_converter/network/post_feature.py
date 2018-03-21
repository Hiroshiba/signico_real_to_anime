import chainer

from deep_image_converter import utility


class DeconvVector(chainer.Chain):
    def __init__(self, out_channels: int, ksize: int):
        super().__init__(
            deconv=chainer.links.Deconvolution2D(None, out_channels, ksize, ksize, 0),
        )

    def __call__(self, x, test):
        return chainer.functions.leaky_relu(self.deconv(x))


class FirstDeconvVector(DeconvVector):
    def __call__(self, x, test):
        assert isinstance(x, list) or isinstance(x, tuple)
        x = list(x)

        h_first = super().__call__(x=x[0], test=test)
        h_other = x[1:]
        return [h_first] + h_other


class MultiDeconvVector(chainer.Chain):
    def __init__(self, out_channels_list, ksize_list):
        super().__init__()
        for i, (out_channels, ksize) in enumerate(zip(out_channels_list, ksize_list)):
            if ksize is not None:
                self.add_link('deconv{}'.format(i + 1), DeconvVector(out_channels, ksize))

        self.num = len(out_channels_list)

    def __call__(self, x, test):
        assert isinstance(x, list) and len(x) == self.num

        h_list = []
        for i, h in enumerate(x):
            key = 'deconv{}'.format(i + 1)
            if not hasattr(self, key):
                h_list.append(None)
                continue

            deconv = getattr(self, key)
            h_list.append(deconv(h, test))
        return h_list
