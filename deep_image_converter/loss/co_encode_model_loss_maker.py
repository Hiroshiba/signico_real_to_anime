import chainer
import numpy

from deep_image_converter.model import CoEncodeModel, BaseDiscriminator
from deep_image_converter.config import LossConfig
from deep_image_converter import utility
from .base_loss import BaseLoss


class CoEncodeModelLossMaker(BaseLoss):
    def __init__(
            self,
            config: LossConfig,
            model: CoEncodeModel,
            dis_a: BaseDiscriminator = None,
            dis_b: BaseDiscriminator = None,
    ):
        super().__init__(config, model)
        self.dis_a = dis_a
        self.dis_b = dis_b

    def _forward_discriminator(self, dis: BaseDiscriminator, real, fake_ae, fake_conv1, fake_conv2, test):
        """
        realの画像ドメインと、fakeの画像ドメインを揃える必要がある
        :param real: 真の画像
        :param fake_ae: オートエンコードにより生成された画像
        :param fake_conv1: １回の変換により生成された画像
        :param fake_conv2: ２回の変換により生成された画像
        """
        return {
            'real': dis(real, test=test)[0],
            'ae': dis(fake_ae, test=test)[0],
            'conv1': dis(fake_conv1, test=test)[0],
            'conv2': dis(fake_conv2, test=test)[0],
        }

    def forward(self, image_a, image_b, test):
        outputs = {}

        for _image, _from, _to, _from_dis, _to_dis \
                in zip([image_a, image_b], ['a', 'b'], ['b', 'a'], [self.dis_a, self.dis_b], [self.dis_b, self.dis_a]):
            h, encoded1 = self.model.encode(_image, _from, test)
            _, decoded_self = self.model.decode(h, _from, test)
            image, decoded1 = self.model.decode(h, _to, test)

            h = image.data
            h, encoded2 = self.model.encode(h, _to, test)
            h, decoded2 = self.model.decode(h, _from, test)

            outputs['{}to{}'.format(_from, _to)] = {
                'input_image': _image,
                'encoded1': encoded1,
                'decoded1': decoded1,
                'decoded_self': decoded_self,
                'encoded2': encoded2,
                'decoded2': decoded2,
            }

        # discriminator forward
        for _image, _target, _other, _dis \
                in zip([image_a, image_b], ['a', 'b'], ['b', 'a'], [self.dis_a, self.dis_b]):
            if _dis is None:
                _output = None
            else:
                fake_ae = outputs['{}to{}'.format(_target, _other)]['decoded_self']['image']
                fake_conv1 = outputs['{}to{}'.format(_other, _target)]['decoded1']['image']
                fake_conv2 = outputs['{}to{}'.format(_target, _other)]['decoded2']['image']
                _output = self._forward_discriminator(_dis, _image, fake_ae, fake_conv1, fake_conv2, test=test)

            outputs['dis_{}'.format(_target)] = _output

        return outputs

    def get_loss_names(self):
        names = ['sum_loss']

        _names = [
            'mse_auto_encoded',
            'mse_twice_converted',
            'mse_z',
        ]
        names += ['{}/{}'.format(key, name) for key in ('atob', 'btoa') for name in _names]

        if self.dis_a is not None:
            _names = [
                'dis_ae',
                'dis_conv1',
                'dis_conv2',
            ]
            names += ['{}/{}'.format(key, name) for key in ('dis_a', 'dis_b') for name in _names]

        return names

    def get_loss_names_discriminator(self):
        return [
            'real',
            'fake_ae',
            'fake_conv1',
            'fake_conv2',
            'accuracy',
            'precision',
            'recall',
        ]

    def make_loss(self, outputs):
        f_mse = chainer.functions.mean_squared_error
        f_lsm = utility.chainer.least_square_mean

        loss = {}

        for from_to in ('atob', 'btoa'):
            loss_part = {}
            output = outputs[from_to]

            input_image = output['input_image']
            encoded1 = output['encoded1']
            decoded1 = output['decoded1']
            decoded_self = output['decoded_self']
            encoded2 = output['encoded2']
            decoded2 = output['decoded2']

            loss_part['mse_auto_encoded'] = f_mse(input_image, decoded_self['image'])
            loss_part['mse_twice_converted'] = f_mse(input_image, decoded2['image'])

            loss_part['mse_z'] = f_mse(encoded1['feature'], encoded2['feature'])

            loss[from_to] = loss_part

        if outputs['dis_a'] is not None:
            for dis_name in ('dis_a', 'dis_b'):
                loss[dis_name] = {}
                output_dis = outputs[dis_name]
                for gen_name in ('ae', 'conv1', 'conv2'):
                    gen_output = output_dis[gen_name]
                    loss[dis_name][gen_name] = \
                        f_lsm(gen_output, self.model.xp.ones(gen_output.shape[0], dtype=numpy.float32))

        for key in loss.keys():
            renamed = {'{}/{}'.format(key, k): v for k, v in loss[key].items()}
            chainer.report(renamed, self.model)

        return loss

    def make_loss_discriminator(self, outputs, a_b: str):
        f_lsm = utility.chainer.least_square_mean

        loss = {}
        dis_model = {'a': self.dis_a, 'b': self.dis_b}[a_b]

        dis_name = 'dis_{}'.format(a_b)
        output_dis = outputs[dis_name]

        output_real = output_dis['real']
        output_fake_ae = output_dis['ae']
        output_fake_conv1 = output_dis['conv1']
        output_fake_conv2 = output_dis['conv2']
        batchsize = output_real.shape[0]

        loss['real'] = f_lsm(output_real, dis_model.xp.ones(batchsize, dtype=numpy.float32))
        loss['fake_ae'] = f_lsm(output_fake_ae, dis_model.xp.zeros(batchsize, dtype=numpy.float32))
        loss['fake_conv1'] = f_lsm(output_fake_conv1, dis_model.xp.zeros(batchsize, dtype=numpy.float32))
        loss['fake_conv2'] = f_lsm(output_fake_conv2, dis_model.xp.zeros(batchsize, dtype=numpy.float32))
        chainer.report(loss, dis_model)

        # accuracy
        output_fake = chainer.functions.concat((output_fake_ae, output_fake_conv1, output_fake_conv2), axis=0)
        tp = (output_real.data > 0.5).sum()
        fp = (output_fake.data > 0.5).sum()
        fn = (output_real.data <= 0.5).sum()
        tn = (output_fake.data <= 0.5).sum()
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if not dis_model.xp.isfinite(precision):
            precision = dis_model.xp.zeros(1, dtype=numpy.float32)

        value = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
        }
        chainer.report(value, dis_model)

        return loss

    def sum_loss(self, loss):
        sum_loss = BaseLoss.blend_loss(loss['atob'], self.config.blend['main'])
        sum_loss += BaseLoss.blend_loss(loss['btoa'], self.config.blend['main'])
        if 'dis_a' in loss:
            sum_loss += BaseLoss.blend_loss(loss['dis_a'], self.config.blend['discriminator'])
            sum_loss += BaseLoss.blend_loss(loss['dis_b'], self.config.blend['discriminator'])
        chainer.report({'sum_loss': sum_loss}, self.model)
        return sum_loss

    def sum_loss_discriminator(self, loss_dis, a_b: str):
        dis_model = {'a': self.dis_a, 'b': self.dis_b}[a_b]
        sum_loss = BaseLoss.blend_loss(loss_dis, self.config.blend_discriminator)
        chainer.report({'sum_loss': sum_loss}, dis_model)
        return sum_loss

    def test(self, image_a, image_b):
        outputs = self.forward(image_a, image_b, test=True)

        # for chainer.report
        if self.dis_a is not None:
            loss_dis = self.make_loss_discriminator(outputs, 'a')
            self.sum_loss_discriminator(loss_dis, 'a')
            loss_dis = self.make_loss_discriminator(outputs, 'b')
            self.sum_loss_discriminator(loss_dis, 'b')

        loss = self.make_loss(outputs)
        return self.sum_loss(loss)
