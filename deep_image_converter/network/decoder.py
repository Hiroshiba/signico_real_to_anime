import chainer

from deep_image_converter import utility

from .deep_deconv import DeepDeconv
from .deep_residual import DeepResidual
from .deep_residual import DeepDilateResidual


class Decoder(chainer.Chain):
    def __init__(self, num_z_residual, num_residual):
        super().__init__(
            residual=DeepResidual(num_z=num_z_residual, num_residual=num_residual),
            deconvolution=DeepDeconv(),
        )

    def __call__(self, x, test):
        h = x
        h = self.residual(h, test)
        h = self.deconvolution(h, test)
        return h


class PoorDilateDecoder(chainer.Chain):
    def __init__(self, num_z_residual, num_residual):
        super().__init__(
            conv=utility.chainer.Convolution2D(None, num_z_residual, 3, 1, 1),
            bn=chainer.links.BatchNormalization(num_z_residual),

            residual=DeepDilateResidual(num_z=num_z_residual, num_residual=num_residual),

            deconv1=utility.chainer.Deconvolution2D(None, num_z_residual, 4, 2, 1),
            bn1=chainer.links.BatchNormalization(num_z_residual),
            deconv2=utility.chainer.Deconvolution2D(None, num_z_residual // 2, 4, 2, 1),
            bn2=chainer.links.BatchNormalization(num_z_residual // 2),
        )

    def __call__(self, x, test):
        h = x
        h = chainer.functions.leaky_relu(self.bn(self.conv(h), test=test))
        h = self.residual(h, test)
        h = chainer.functions.leaky_relu(self.bn1(self.deconv1(h), test=test))
        h = chainer.functions.leaky_relu(self.bn2(self.deconv2(h), test=test))
        return h
