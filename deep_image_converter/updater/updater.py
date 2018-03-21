import chainer
import typing

from deep_image_converter.loss import *
from deep_image_converter import utility


class AutoEncoderUpdater(chainer.training.StandardUpdater):
    _TargetLossMaker = typing.Union[AutoEncoderLossMaker, VariationalAutoEncoderLossMaker]

    def __init__(self, loss_maker: _TargetLossMaker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_maker = loss_maker

    def update_core(self):
        optimizers = self.get_all_optimizers()

        batch = self.converter(self.get_iterator('main').next(), self.device)

        outputs = self.loss_maker.forward(chainer.Variable(batch['input']), test=False)

        loss = self.loss_maker.make_loss(outputs, chainer.Variable(batch['target']))
        optimizers['main'].update(self.loss_maker.sum_loss, loss)

        if 'dis' in optimizers:
            loss = self.loss_maker.make_loss_discriminator(outputs)
            optimizers['dis'].update(self.loss_maker.sum_loss_discriminator, loss)

        if 'ldis' in optimizers:
            loss = self.loss_maker.make_loss_latent_discriminator(outputs)
            optimizers['ldis'].update(self.loss_maker.sum_loss_latent_discriminator, loss)


class ConvertModelUpdater(chainer.training.StandardUpdater):
    _TargetLossMaker = typing.Union[ConvertModelLossMaker, FacebookConvertModelLossMaker]

    def __init__(self, loss_maker: _TargetLossMaker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_maker = loss_maker

    def update_core(self):
        optimizers = self.get_all_optimizers()

        batch_a = chainer.Variable(self.converter(self.get_iterator('a').next(), self.device))
        batch_b = chainer.Variable(self.converter(self.get_iterator('b').next(), self.device))

        outputs = self.loss_maker.forward(batch_a, batch_b, test=False)

        loss = self.loss_maker.make_loss(outputs)
        optimizers['main'].update(self.loss_maker.sum_loss, loss)

        loss = self.loss_maker.make_loss_discriminator(outputs)
        optimizers['dis'].update(self.loss_maker.sum_loss_discriminator, loss)

    @property
    def epoch(self):
        return 0

    @property
    def epoch_detail(self):
        return 0


class CoEncodeModelUpdater(chainer.training.StandardUpdater):
    def __init__(self, loss_maker: CoEncodeModelLossMaker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_maker = loss_maker

    def update_core(self):
        optimizers = self.get_all_optimizers()

        batch_a = utility.chainer.to_variable_recursive(self.converter(self.get_iterator('a').next(), self.device))
        batch_b = utility.chainer.to_variable_recursive(self.converter(self.get_iterator('b').next(), self.device))

        outputs = self.loss_maker.forward(batch_a['input'], batch_b['input'], test=False)

        loss = self.loss_maker.make_loss(outputs)
        optimizers['main'].update(self.loss_maker.sum_loss, loss)

        for a_b in ('a', 'b'):
            key = 'dis_{}'.format(a_b)
            if key not in optimizers.keys():
                continue

            loss = self.loss_maker.make_loss_discriminator(outputs, a_b)
            optimizers[key].update(self.loss_maker.sum_loss_discriminator, loss, a_b)

    @property
    def epoch(self):
        return 0

    @property
    def epoch_detail(self):
        return 0
