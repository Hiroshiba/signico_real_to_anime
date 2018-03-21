from abc import ABCMeta, abstractmethod
import chainer

from deep_image_converter.config import LossConfig


class BaseLoss(object, metaclass=ABCMeta):
    def __init__(self, config: LossConfig, model):
        self.config = config
        self.model = model

    @staticmethod
    def blend_loss(loss, blend_config):
        assert sorted(loss.keys()) == sorted(blend_config.keys()), '{} {}'.format(loss.keys(), blend_config.keys())

        sum_loss = None

        for key in sorted(loss.keys()):
            blend = blend_config[key]
            if blend == 0.0:
                continue

            l = loss[key] * blend_config[key]

            if sum_loss is None:
                sum_loss = l
            else:
                sum_loss += l

        return sum_loss

    def should_compute(self, key: str):
        return key in self.config.blend['main']

    def get_loss_names(self):
        return ['sum_loss'] + list(self.config.blend['main'].keys())

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def make_loss(self, outputs, target):
        pass

    @abstractmethod
    def sum_loss(self, loss):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass
