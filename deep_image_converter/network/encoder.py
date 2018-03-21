import chainer

from deep_image_converter import utility

from .deep_conv import DeepConv
from .deep_residual import DeepResidual
from .deep_residual import DeepDilateResidual


class Encoder(chainer.Chain):
    def __init__(self, num_z_residual, num_residual):
        super().__init__(
            convolution=DeepConv(),
            residual=DeepResidual(num_z=num_z_residual, num_residual=num_residual),
        )

    def __call__(self, x, test):
        h = x
        h = self.convolution(h, test)
        h = self.residual(h, test)
        return h


class PoorDilateEncoder(chainer.Chain):
    def __init__(self, num_z_residual, num_residual):
        super().__init__(
            conv=utility.chainer.Convolution2D(None, num_z_residual, 3, 1, 1),
            bn=chainer.links.BatchNormalization(num_z_residual),
            residual=DeepDilateResidual(num_z=num_z_residual, num_residual=num_residual),
        )

    def __call__(self, x, test):
        h = x
        h = chainer.functions.average_pooling_2d(h, 4, stride=4)
        h = chainer.functions.leaky_relu(self.bn(self.conv(h), test))
        h = self.residual(h, test)
        return h
