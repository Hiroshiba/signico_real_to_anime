import argparse
import chainer
import glob
import os
import sys

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT_PATH)

from deep_image_converter.config import Config
from deep_image_converter import dataset
from deep_image_converter.loss import ConvertModelLossMaker, FacebookConvertModelLossMaker
from deep_image_converter.model import prepare_model, choose_discriminator, BaseConvertModel
from deep_image_converter.updater import ConvertModelUpdater
from deep_image_converter.train import TrainManager
from deep_image_converter import utility

parser = argparse.ArgumentParser()
parser.add_argument('config_json_path')
config_json_path = parser.parse_args().config_json_path

config = Config(config_json_path)
config.copy_config_json()

train_manager = TrainManager(config.train_config)

datasets = dataset.choose(config.dataset_config)

nb = config.train_config.batchsize

IteratorClass = chainer.iterators.MultiprocessIterator
iterator_train_a = IteratorClass(datasets['train_a'], nb, True, True)
iterator_train_b = IteratorClass(datasets['train_b'], nb, True, True)
iterator_test = IteratorClass(datasets['test'], nb, False, False)
iterator_train_eval = IteratorClass(datasets['train_eval'], nb, False, False)

config.train_config.gpu >= 0 and chainer.cuda.get_device(config.train_config.gpu).use()

utility.chainer.set_default_initialW(config.model_config.initialW)

model = prepare_model(config.model_config)
assert isinstance(model, BaseConvertModel)
config.train_config.gpu >= 0 and model.to_gpu()

optimizer = train_manager.make_optimizer(model, 'main')
optimizers = {'main': optimizer}

dis = choose_discriminator(config.model_config)
config.train_config.gpu >= 0 and dis.to_gpu()

optimizer = train_manager.make_optimizer(dis, 'discriminator')
optimizers['dis'] = optimizer

if config.loss_config.name is None:
    loss_maker = ConvertModelLossMaker(config.loss_config, model, dis)
elif config.loss_config.name == 'facebook':
    loss_maker = FacebookConvertModelLossMaker(config.loss_config, model, dis)
else:
    raise NotImplementedError(config.loss_config.name)


updater = ConvertModelUpdater(
    optimizer=optimizers,
    iterator={'a': iterator_train_a, 'b': iterator_train_b},
    loss_maker=loss_maker,
    device=config.train_config.gpu,
)

trainer = train_manager.make_trainer(
    updater=updater,
    model={'main': model, 'dis': dis},
    eval_func=loss_maker.test,
    iterator_test=iterator_test,
    iterator_train_eval=iterator_train_eval,
    loss_names=loss_maker.get_loss_names() + loss_maker.get_loss_names_discriminator(),
)
trainer.run()
