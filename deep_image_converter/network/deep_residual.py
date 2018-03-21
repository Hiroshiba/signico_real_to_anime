import chainer

from .residual_block import ResidualBlock, DilateResidualBlock


class DeepResidual(chainer.Chain):
    def __init__(self, num_z, num_residual):
        super().__init__()
        self.num_residual = num_residual

        for i in range(num_residual):
            self.add_link('res{}'.format(i), ResidualBlock(num_z))

    def __call__(self, x, test):
        h = x
        for i in range(self.num_residual):
            h = getattr(self, 'res{}'.format(i))(h, test)
        return h


class DeepDilateResidual(chainer.Chain):
    def __init__(self, num_z, num_residual):
        super().__init__()
        self.num_residual = num_residual

        for i in range(num_residual):
            dilate = 2 ** (i + 1)
            self.add_link('res{}'.format(i), DilateResidualBlock(num_z, dilate))

    def __call__(self, x, test):
        h = x
        for i in range(self.num_residual):
            h = getattr(self, 'res{}'.format(i))(h, test)
        return h
