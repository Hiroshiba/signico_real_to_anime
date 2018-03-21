import chainer
import numpy

from deep_image_converter.model import BaseAutoEncoderModel, BaseDiscriminator
from deep_image_converter.config import LossConfig
from deep_image_converter import utility
from .base_loss import BaseLoss


class AutoEncoderLossMaker(BaseLoss):
    def __init__(
            self,
            config: LossConfig,
            model: BaseAutoEncoderModel,
            dis: BaseDiscriminator,
            ldis: BaseDiscriminator,
    ):
        super().__init__(config, model)
        self.dis = dis
        self.ldis = ldis

    def _forward_discriminator(self, dis: BaseDiscriminator, real, fake, test):
        return {
            'real': dis(real, test=test)[0],
            'fake': dis(fake, test=test)[0],
        }

    def forward(self, image, test):
        latent_fake, encoded = self.model.encode(image, test)
        image_fake, decoded = self.model.decode(latent_fake, test)

        outputs = {
            'encoded': encoded,
            'decoded': decoded,
        }

        # discriminator forward
        if self.dis is not None:
            outputs['dis'] = self._forward_discriminator(self.dis, image, image_fake, test=test)

        if self.ldis is not None:
            latent_real = self.model.get_random_feature(batchsize=len(latent_fake))
            outputs['ldis'] = self._forward_discriminator(self.ldis, latent_real, latent_fake, test=test)

        return outputs

    def get_loss_names_discriminator(self):
        return [
            'real',
            'fake',
        ]

    def make_loss(self, outputs, target):
        f_mse = chainer.functions.mean_squared_error
        f_nll = chainer.functions.gaussian_nll
        f_bnll = chainer.functions.bernoulli_nll
        f_softmax_ce = chainer.functions.softmax_cross_entropy
        f_lsm = utility.chainer.least_square_mean

        loss = {}

        target_image = target
        feature = outputs['encoded']['feature']
        output_image = outputs['decoded']['image']

        if self.should_compute('mse_auto_encoded'):
            loss['mse_auto_encoded'] = f_mse(target_image, output_image)

        if self.should_compute('sce_auto_encoded'):
            loss['sce_auto_encoded'] = f_softmax_ce(output_image, target_image)

        if self.should_compute('nll_auto_encoded'):
            num_feature = outputs['decoded']['mu'].size
            loss['nll_auto_encoded'] = f_nll(
                target,
                outputs['decoded']['mu'],
                outputs['decoded']['var'],
            ) / num_feature

        if self.should_compute('bnll_auto_encoded'):
            num_feature = outputs['decoded']['b'].size
            loss['bnll_auto_encoded'] = f_bnll(
                target,
                outputs['decoded']['b'],
            ) / num_feature

        if self.should_compute('standard_gaussian_feature'):
            loss['standard_gaussian_feature'] = utility.chainer.standard_gaussian_loss(feature)

        if self.dis is not None:
            gen_output = outputs['dis']['fake']
            loss['discriminator'] = f_lsm(gen_output, self.model.xp.ones(gen_output.shape[0], dtype=numpy.float32))

        if self.ldis is not None:
            gen_output = outputs['ldis']['fake']
            print(gen_output.shape)
            loss['latent_discriminator'] = f_lsm(gen_output, self.model.xp.ones(gen_output.shape[0], dtype=numpy.float32))

        chainer.report(loss, self.model)

        return loss

    @staticmethod
    def _make_loss_discriminator(outputs, key: str, model: BaseDiscriminator):
        f_lsm = utility.chainer.least_square_mean

        output_real, output_fake = outputs[key]['real'], outputs[key]['fake']
        loss = {
            'real': f_lsm(output_real, model.xp.ones(output_real.shape[0], dtype=numpy.float32)),
            'fake': f_lsm(output_fake, model.xp.zeros(output_fake.shape[0], dtype=numpy.float32)),
        }

        chainer.report(loss, model)
        return loss

    def make_loss_discriminator(self, outputs):
        return AutoEncoderLossMaker._make_loss_discriminator(outputs, 'dis', self.dis)

    def make_loss_latent_discriminator(self, outputs):
        return AutoEncoderLossMaker._make_loss_discriminator(outputs, 'ldis', self.ldis)

    def sum_loss(self, loss):
        sum_loss = BaseLoss.blend_loss(loss, self.config.blend['main'])
        chainer.report({'sum_loss': sum_loss}, self.model)
        return sum_loss

    def sum_loss_discriminator(self, loss_dis):
        sum_loss = BaseLoss.blend_loss(loss_dis, self.config.blend_discriminator)
        chainer.report({'sum_loss': sum_loss}, self.dis)
        return sum_loss

    def sum_loss_latent_discriminator(self, loss_dis):
        sum_loss = BaseLoss.blend_loss(loss_dis, self.config.blend_latent_discriminator)
        chainer.report({'sum_loss': sum_loss}, self.ldis)
        return sum_loss

    def test(self, input, target):
        outputs = self.forward(input, test=True)

        # for chainer.report
        if self.dis is not None:
            loss_dis = self.make_loss_discriminator(outputs)
            self.sum_loss_discriminator(loss_dis)

        if self.ldis is not None:
            loss_dis = self.make_loss_latent_discriminator(outputs)
            self.sum_loss_latent_discriminator(loss_dis)

        loss = self.make_loss(outputs, target)
        return self.sum_loss(loss)


class VariationalAutoEncoderLossMaker(AutoEncoderLossMaker):
    def make_loss(self, outputs, target):
        loss = super().make_loss(outputs, target)

        f_kl = chainer.functions.gaussian_kl_divergence
        f_gkl = utility.chainer.gradation_gaussian_kl_divergence

        num_feature = outputs['encoded']['mu'].size
        if self.should_compute('kl'):
            loss['kl'] = f_kl(outputs['encoded']['mu'], outputs['encoded']['var']) / num_feature

        if self.should_compute('kl_gradation'):
            loss['kl_gradation'] = f_gkl(
                outputs['encoded']['mu'],
                outputs['encoded']['var'],
                self.config.other['kl_gradation_min'],
                self.config.other['kl_gradation_max'],
            ) / num_feature

        chainer.report(loss, self.model)

        return loss
