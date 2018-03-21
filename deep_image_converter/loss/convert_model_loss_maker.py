import chainer
import numpy

from deep_image_converter.model import BaseConvertModel, BaseDiscriminator
from deep_image_converter.config import LossConfig
from deep_image_converter import utility
from .base_loss import BaseLoss


class ConvertModelLossMaker(BaseLoss):
    def __init__(
            self,
            config: LossConfig,
            model: BaseConvertModel,
            dis: BaseDiscriminator,
    ):
        super().__init__(config, model)
        self.dis = dis

    def _forward_discriminator(self, dis: BaseDiscriminator, real, fake, test):
        """
        realの画像ドメインと、fakeの画像ドメインを揃える必要がある
        """
        return {
            'real': dis(real, test=test)[0],
            'fake': dis(fake, test=test)[0],
        }

    def forward(self, image_a, image_b, test):
        h, encoded = self.model.encode(image_a, test)
        image_fake, decoded = self.model.decode(h, test)

        outputs = {
            'encoded': encoded,
            'decoded': decoded,
        }

        # discriminator forward
        outputs['dis'] = self._forward_discriminator(self.dis, image_b, image_fake, test=test)

        return outputs

    def get_loss_names(self):
        return ['sum_loss', 'discriminator']

    def get_loss_names_discriminator(self):
        return [
            'real',
            'fake',
        ]

    def make_loss(self, outputs):
        f_lsm = utility.chainer.least_square_mean

        loss = {}

        # encoded = outputs['encoded']
        # decoded = outputs['decoded']

        gen_output = outputs['dis']['fake']
        loss['discriminator'] = f_lsm(gen_output, self.model.xp.ones(gen_output.shape[0], dtype=numpy.float32))

        chainer.report(loss, self.model)

        return loss

    def make_loss_discriminator(self, outputs):
        f_lsm = utility.chainer.least_square_mean

        loss = {}

        output_real = outputs['dis']['real']
        loss['real'] = f_lsm(output_real, self.dis.xp.ones(output_real.shape[0], dtype=numpy.float32))

        fake_output = outputs['dis']['fake']
        loss['fake'] = f_lsm(fake_output, self.dis.xp.zeros(fake_output.shape[0], dtype=numpy.float32))

        chainer.report(loss, self.dis)

        return loss

    def sum_loss(self, loss):
        sum_loss = BaseLoss.blend_loss(loss, self.config.blend['main'])
        chainer.report({'sum_loss': sum_loss}, self.model)
        return sum_loss

    def sum_loss_discriminator(self, loss_dis):
        sum_loss = BaseLoss.blend_loss(loss_dis, self.config.blend_discriminator)
        chainer.report({'sum_loss': sum_loss}, self.dis)
        return sum_loss

    def test(self, image_a, image_b):
        outputs = self.forward(image_a, image_b, test=True)

        # for chainer.report
        loss_dis = self.make_loss_discriminator(outputs)
        self.sum_loss_discriminator(loss_dis)

        loss = self.make_loss(outputs)
        return self.sum_loss(loss)
