import chainer
import numpy

from deep_image_converter.model import BaseConvertModel, BaseDiscriminator
from deep_image_converter.config import LossConfig
from .base_loss import BaseLoss


class FacebookConvertModelLossMaker(BaseLoss):
    def __init__(
            self,
            config: LossConfig,
            model: BaseConvertModel,
            dis: BaseDiscriminator,
    ):
        super().__init__(config, model)
        self.dis = dis

    def _forward_discriminator(self, dis: BaseDiscriminator, real, fake_ae, fake_conv, test):
        """
        realの画像ドメインと、fakeの画像ドメインを揃える必要がある
        """
        return {
            'real': dis(real, test=test)[0],
            'ae': dis(fake_ae, test=test)[0],
            'conv': dis(fake_conv, test=test)[0],
        }

    def forward(self, image_a, image_b, test):
        h, encoded1 = self.model.encode(image_a, test)
        image_conv, decoded = self.model.decode(h, test)

        h = image_conv.data
        h, encoded2 = self.model.encode(h, test)

        h, _ = self.model.encode(image_b, test)
        image_ae, decoded_self = self.model.decode(h, test)

        outputs = {
            'image_b': image_b,
            'encoded1': encoded1,
            'decoded_self': decoded_self,
            'decoded': decoded,
            'encoded2': encoded2,
        }

        # discriminator forward
        outputs['dis'] = self._forward_discriminator(self.dis, image_b, image_ae, image_conv, test=test)

        return outputs

    def get_loss_names(self):
        return ['sum_loss', 'discriminator']

    def get_loss_names_discriminator(self):
        return [
            'real',
            'fake',
        ]

    def make_loss(self, outputs):
        f_mse = chainer.functions.mean_squared_error
        f_sce = chainer.functions.sigmoid_cross_entropy

        loss = {}

        image_b = outputs['image_b']
        encoded1 = outputs['encoded1']
        decoded_self = outputs['decoded_self']
        encoded2 = outputs['encoded2']

        loss['main'] = {
            'mse_auto_encoded': f_mse(image_b, decoded_self['image']),
            'mse_z': f_mse(encoded1['encoded'], encoded2['encoded']),
        }

        gen_output_ae = outputs['dis']['ae']
        gen_output_conv = outputs['dis']['conv']
        loss['dis'] = {
            'ae': f_sce(gen_output_ae, self.model.xp.ones(gen_output_ae.shape[0], dtype=numpy.int32)),
            'conv': f_sce(gen_output_conv, self.model.xp.ones(gen_output_conv.shape[0], dtype=numpy.int32)),
        }

        chainer.report(loss, self.model)

        return loss

    def make_loss_discriminator(self, outputs):
        f_sce = chainer.functions.sigmoid_cross_entropy

        loss = {}

        output_real = outputs['dis']['real']
        output_fake_ae = outputs['dis']['ae']
        output_fake_conv = outputs['dis']['conv']
        batchsize = output_real.shape[0]

        loss['real'] = f_sce(output_real, self.dis.xp.ones(batchsize, dtype=numpy.int32))
        loss['fake_ae'] = f_sce(output_fake_ae, self.dis.xp.zeros(batchsize, dtype=numpy.int32))
        loss['fake_conv'] = f_sce(output_fake_conv, self.dis.xp.zeros(batchsize, dtype=numpy.int32))
        chainer.report(loss, self.dis)

        # accuracy
        output_fake = chainer.functions.concat((output_fake_ae, output_fake_conv), axis=0)
        tp = (output_real.data > 0.5).sum()
        fp = (output_fake.data > 0.5).sum()
        fn = (output_real.data <= 0.5).sum()
        tn = (output_fake.data <= 0.5).sum()
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if not self.dis.xp.isfinite(precision):
            precision = self.dis.xp.zeros(1, dtype=numpy.float32)

        value = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
        }
        chainer.report(value, self.dis)

        return loss

    def sum_loss(self, loss):
        sum_loss = BaseLoss.blend_loss(loss['main'], self.config.blend['main'])
        sum_loss += BaseLoss.blend_loss(loss['dis'], self.config.blend['discriminator'])
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
