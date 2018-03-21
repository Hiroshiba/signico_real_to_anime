import chainer
from chainer import cuda

_default_initialW = None


def to_device(elem: chainer.Variable, device=None):
    if device is None:
        return elem

    elif device < 0:
        elem.to_cpu()

    else:
        elem.to_gpu(device=device)


def to_variable(elem, device=None):
    if elem is None:
        return None
    elif isinstance(elem, chainer.Variable):
        pass
    else:
        elem = chainer.Variable(elem)

    to_device(elem, device)
    return elem


def to_variable_recursive(obj, device=None):
    if isinstance(obj, tuple) or isinstance(obj, list):
        return [to_variable_recursive(elem, device) for elem in obj]

    elif isinstance(obj, dict):
        return {key: to_variable_recursive(obj[key], device) for key in obj}

    else:
        return to_variable(obj, device)


def concat_recursive(batch):
    first = batch[0]

    if isinstance(first, tuple):
        return tuple([concat_recursive([example[i] for example in batch]) for i in range(len(first))])

    elif isinstance(first, dict):
        return {key: concat_recursive([example[key] for example in batch]) for key in first}

    else:
        return _concat_arrays(batch)


def _concat_arrays(arrays):
    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device(arrays[0]):
        return xp.concatenate([array[None] for array in arrays])


def standard_gaussian_loss(x):
    bn = x.shape[0]
    mean = chainer.functions.sum(x, axis=0) / bn
    power = chainer.functions.sum(x * x, axis=0)
    # var = chainer.functions.sum((x - mean) ** 2) / x.size
    var = power / bn - mean * mean
    loss = (power / bn - chainer.functions.log(var) - 1) * 0.5
    return chainer.functions.sum(loss) / loss.size
    # return (mean * mean + var - chainer.functions.log(var) - 1) * 0.5


def gradation_gaussian_kl_divergence(mean, ln_var, ming: float, maxg: float):
    xp = chainer.cuda.get_array_module(mean.data)
    _sum = chainer.functions.sum

    bn = mean.shape[0]
    J = mean.size
    var = chainer.functions.exp(ln_var)

    a = xp.logspace(xp.log10(ming), xp.log10(maxg), J // bn)
    a = a.reshape(mean.shape[1:])
    return _sum(a * (_sum(mean * mean, axis=0) + _sum(var, axis=0) - _sum(ln_var, axis=0) - bn)) * 0.5


def least_square_mean(x, y):
    return chainer.functions.sum((x - y) ** 2) / (2 * x.size)


class ChainList(chainer.ChainList):
    def __init__(self, *links, forwarder=None):
        super().__init__(*links)
        self.forwarder = forwarder

    def __call__(self, h, *args, **kwargs):
        if self.forwarder is None:
            for child in self._children:
                h = child(h, *args, **kwargs)
        else:
            for i, (child, forwarder) in enumerate(zip(self._children, self.forwarder)):
                h = forwarder(h, i, child, *args, **kwargs)

        return h


class FunctionLink(chainer.Link):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def __call__(self, h, *args, **kwargs):
        return self.function(h)


def set_default_initialW(initialW):
    global _default_initialW

    if isinstance(initialW, str):
        if initialW == 'Orthogonal':
            initialW = chainer.initializers.Orthogonal()

    _default_initialW = initialW


class Convolution2D(chainer.links.Convolution2D):
    def __init__(self, *args, **kwargs):
        initialW = kwargs.pop('initialW', _default_initialW)
        super().__init__(*args, initialW=initialW, **kwargs)


class Deconvolution2D(chainer.links.Deconvolution2D):
    def __init__(self, *args, **kwargs):
        initialW = kwargs.pop('initialW', _default_initialW)
        super().__init__(*args, initialW=initialW, **kwargs)


class DilatedConvolution2D(chainer.links.DilatedConvolution2D):
    def __init__(self, *args, **kwargs):
        print(_default_initialW)
        initialW = kwargs.pop('initialW', _default_initialW)
        super().__init__(*args, initialW=initialW, **kwargs)


class Linear(chainer.links.Linear):
    def __init__(self, *args, **kwargs):
        initialW = kwargs.pop('initialW', _default_initialW)
        super().__init__(*args, initialW=initialW, **kwargs)
