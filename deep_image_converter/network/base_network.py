from abc import ABCMeta, abstractmethod
import chainer


class BaseNetwork(chainer.Chain, metaclass=ABCMeta):
    def __init__(self, base_activation=None, last_activation=None, **kwargs):
        super().__init__(**kwargs)

        self.base_activation = base_activation if base_activation is not None else chainer.functions.leaky_relu
        self.last_activation = last_activation if last_activation is not None else self.base_activation

    @abstractmethod
    def __call__(self, x, test: bool):
        pass

    def activation(self, h, is_last=False):
        if not is_last:
            return self.base_activation(h)
        else:
            return self.last_activation(h)
