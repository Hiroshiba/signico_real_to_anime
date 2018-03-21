import chainer
import numpy

from deep_image_converter import network
from deep_image_converter.config import ModelConfig

from .base_model import BaseEncoder


class ConvFeatureEncoder(BaseEncoder):
    def __init__(
            self,
            config: ModelConfig,
            encode_network=None,
    ):
        super().__init__(
            config=config,
            encode_network=encode_network,
            feature_network=network.ConvVector(num_z=config.num_z_feature, size=config.image_width // 2 ** 4),
        )

    def get_random_feature(self, batchsize):
        mu = self.xp.zeros((batchsize, self.config.num_z_feature, 1, 1), dtype=numpy.float32)
        var = self.xp.ones((batchsize, self.config.num_z_feature, 1, 1), dtype=numpy.float32)
        z = chainer.functions.gaussian(mu, var)
        return z


class MuVarFeatureEncoder(BaseEncoder):
    def __init__(
            self,
            config: ModelConfig,
            encode_network=None,
    ):
        super().__init__(
            config=config,
            encode_network=encode_network,
            feature_network=network.VaeMuVarVertor(num_z=config.num_z_feature, size=config.image_width // 2 ** 4),
        )

    def __call__(self, x, test):
        h = x
        h, other = super().__call__(h, test)
        other['mu'], other['var'] = h

        sampling = lambda _h: chainer.functions.gaussian(_h[0], _h[1]) if not test else _h[0]

        h = sampling(h)
        other['feature'] = h
        other['sampling'] = sampling
        return h, other

    def get_random_feature(self, batchsize):
        mu = self.xp.zeros((batchsize, self.config.num_z_feature), dtype=numpy.float32)
        var = self.xp.ones((batchsize, self.config.num_z_feature), dtype=numpy.float32)
        z = chainer.functions.gaussian(mu, var)
        z = chainer.functions.reshape(z, (batchsize, self.config.num_z_feature, 1, 1))
        return z


class MultiMuVarFeatureEncoder(BaseEncoder):
    def __init__(
            self,
            config: ModelConfig,
            encode_network,
            num_z_list,
            size_list,
    ):
        super().__init__(
            config=config,
            encode_network=encode_network,
            feature_network=network.MultiVaeMuVarVertor(num_z_list=num_z_list, size_list=size_list)
        )
        self.num_z_list = num_z_list
        self.size_list = size_list

    def __call__(self, x, test):
        h = x
        h_list, other = super().__call__(h, test)

        mu_list = []
        var_list = []
        for h in h_list:
            if h is not None:
                mu_list += [h[0]]
                var_list += [h[1]]
            else:
                mu_list += [None]
                var_list += [None]
        other['mu_list'] = mu_list
        other['var_list'] = var_list

        bn = len(h_list[0])
        other['mu'] = chainer.functions.concat([chainer.functions.reshape(mu, (bn, -1)) for mu in mu_list if mu is not None])
        other['var'] = chainer.functions.concat([chainer.functions.reshape(var, (bn, -1)) for var in var_list if var is not None])

        def sampling(mu_list, var_list):
            if not test:
                h_list = []
                for mu, var in zip(mu_list, var_list):
                    if mu is not None:
                        h_list += [chainer.functions.gaussian(mu, var)]
                    else:
                        h_list += [None]
                return h_list
            else:
                return mu_list

        h = sampling(mu_list, var_list)
        other['feature'] = h
        other['sampling'] = sampling
        return h, other

    def get_random_feature(self, batchsize):
        size_z = 1
        z_list = []
        for num_z, size in zip(self.num_z_list, self.size_list):
            if num_z is not None:
                mu = self.xp.zeros((batchsize, num_z, size_z, size_z), dtype=numpy.float32)
                var = self.xp.ones((batchsize, num_z, size_z, size_z), dtype=numpy.float32)
                z = chainer.functions.gaussian(mu, var)
                z_list += [z]

                size_z *= size
            else:
                z_list += [None]
        return z_list
