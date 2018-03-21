import argparse
import chainer
import glob
import os
import sys

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT_PATH)

from deep_image_converter.config import Config
from deep_image_converter import dataset
from deep_image_converter.loss import AutoEncoderLossMaker, VariationalAutoEncoderLossMaker
from deep_image_converter.model import prepare_model, choose_discriminator, BaseAutoEncoderModel, LatentDiscriminator
from deep_image_converter.updater import AutoEncoderUpdater
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
iterator_train = IteratorClass(datasets['train'], nb, True, True)
iterator_test = IteratorClass(datasets['test'], nb, False, False)
iterator_train_eval = IteratorClass(datasets['train_eval'], nb, False, False)

config.train_config.gpu >= 0 and chainer.cuda.get_device(config.train_config.gpu).use()

utility.chainer.set_default_initialW(config.model_config.initialW)

model = prepare_model(config.model_config)
assert isinstance(model, BaseAutoEncoderModel)
config.train_config.gpu >= 0 and model.to_gpu()
models = {'main': model}

optimizer = train_manager.make_optimizer(model, 'main')
optimizers = {'main': optimizer}

if config.model_config.discriminator is not None:
    dis = choose_discriminator(config.model_config)
    config.train_config.gpu >= 0 and dis.to_gpu()
    models['dis'] = dis

    optimizer = train_manager.make_optimizer(dis, 'discriminator')
    optimizers['dis'] = optimizer
else:
    dis = None

if config.model_config.latent_discriminator is not None:
    ldis = LatentDiscriminator(config.model_config)
    config.train_config.gpu >= 0 and ldis.to_gpu()
    models['ldis'] = ldis

    optimizer = train_manager.make_optimizer(ldis, 'latent_discriminator')
    optimizers['ldis'] = optimizer
else:
    ldis = None

if config.loss_config.name is None or config.loss_config.name == 'ae':
    loss_maker = AutoEncoderLossMaker(config.loss_config, model, dis, ldis)
elif config.loss_config.name == 'vae':
    loss_maker = VariationalAutoEncoderLossMaker(config.loss_config, model, dis, ldis)
else:
    raise NotImplementedError(config.loss_config.name)

updater = AutoEncoderUpdater(
    optimizer=optimizers,
    iterator=iterator_train,
    loss_maker=loss_maker,
    device=config.train_config.gpu,
)

trainer = train_manager.make_trainer(
    updater=updater,
    model=models,
    eval_func=loss_maker.test,
    iterator_test=iterator_test,
    iterator_train_eval=iterator_train_eval,
    loss_names=loss_maker.get_loss_names() + loss_maker.get_loss_names_discriminator(),
)
trainer.run()
