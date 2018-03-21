from abc import ABCMeta, abstractmethod
import chainer
import math
import typing

from deep_image_converter.config import ModelConfig
from deep_image_converter import utility

from .base_model import BaseModel


class BaseDiscriminator(BaseModel, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, x, test) -> (chainer.Variable, typing.Dict):
        pass


class Discriminator(BaseDiscriminator):
    def __init__(self, config: ModelConfig):
        ndf = config.num_z_base_discriminator
        super().__init__(
            config,
            c1=utility.chainer.Convolution2D(None, ndf * 1, 4, 2, 1),
            c2=utility.chainer.Convolution2D(None, ndf * 1, 3, 1, 1),
            c3=utility.chainer.Convolution2D(None, ndf * 2, 4, 2, 1),
            c4=utility.chainer.Convolution2D(None, ndf * 2, 3, 1, 1),
            c5=utility.chainer.Convolution2D(None, ndf * 4, 4, 2, 1),
            c6=utility.chainer.Convolution2D(None, ndf * 4, 3, 1, 1),
            c7=utility.chainer.Convolution2D(None, ndf * 8, 4, 2, 1),
            l8l=utility.chainer.Linear(None, 1),

            bnc1=chainer.links.BatchNormalization(ndf * 1),
            bnc2=chainer.links.BatchNormalization(ndf * 1),
            bnc3=chainer.links.BatchNormalization(ndf * 2),
            bnc4=chainer.links.BatchNormalization(ndf * 2),
            bnc5=chainer.links.BatchNormalization(ndf * 4),
            bnc6=chainer.links.BatchNormalization(ndf * 4),
            bnc7=chainer.links.BatchNormalization(ndf * 8),
        )

    def __call__(self, x, test):
        h = x
        h = chainer.functions.leaky_relu(self.bnc1(self.c1(h), test=test))
        h = chainer.functions.leaky_relu(self.bnc2(self.c2(h), test=test))
        h = chainer.functions.leaky_relu(self.bnc3(self.c3(h), test=test))
        h = chainer.functions.leaky_relu(self.bnc4(self.c4(h), test=test))
        h = chainer.functions.leaky_relu(self.bnc5(self.c5(h), test=test))
        h = chainer.functions.leaky_relu(self.bnc6(self.c6(h), test=test))
        h = chainer.functions.leaky_relu(self.bnc7(self.c7(h), test=test))
        h = chainer.functions.squeeze(self.l8l(h))
        return h, {}


class PoorDiscriminator(BaseDiscriminator):
    def __init__(self, config: ModelConfig):
        ndf = config.num_z_base_discriminator
        super().__init__(
            config,
            c0=utility.chainer.Convolution2D(None, ndf * 1, 4, 2, 1),
            c1=utility.chainer.Convolution2D(None, ndf * 2, 4, 2, 1),
            c2=utility.chainer.Convolution2D(None, ndf * 4, 4, 2, 1),
            c3=utility.chainer.Convolution2D(None, ndf * 8, 4, 2, 1),
            c4=utility.chainer.Convolution2D(None, 1, config.image_width // (2 ** 4)),
            bn1=chainer.links.BatchNormalization(ndf * 2),
            bn2=chainer.links.BatchNormalization(ndf * 4),
            bn3=chainer.links.BatchNormalization(ndf * 8),
        )

    def __call__(self, x, test):
        h = chainer.functions.leaky_relu(self.c0(x))
        h = chainer.functions.leaky_relu(self.bn1(self.c1(h), test=test))
        h = chainer.functions.leaky_relu(self.bn2(self.c2(h), test=test))
        h = chainer.functions.leaky_relu(self.bn3(self.c3(h), test=test))
        h = chainer.functions.squeeze(self.c4(h))
        return h, {}


class LatentDiscriminator(BaseDiscriminator):
    def __init__(self, config: ModelConfig):
        super().__init__(
            config,
            l0=chainer.links.Linear(None, 1000),
            l1=chainer.links.Linear(None, 1000),
            l2=chainer.links.Linear(None, 1),
        )

    def __call__(self, x, test):
        h = x
        h = chainer.functions.leaky_relu(self.l0(h))
        h = chainer.functions.leaky_relu(self.l1(h))
        h = chainer.functions.squeeze(self.l2(h))
        return h, {}
