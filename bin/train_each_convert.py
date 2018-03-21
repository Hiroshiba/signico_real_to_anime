import argparse
import chainer
import glob
import os
import sys

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT_PATH)

from deep_image_converter.config import Config
from deep_image_converter import dataset
from deep_image_converter.loss import CoEncodeModelLossMaker
from deep_image_converter.model import prepare_model, choose_discriminator, BaseEachConvertModel, Discriminator
from deep_image_converter.updater import CoEncodeModelUpdater
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
assert isinstance(model, BaseEachConvertModel)
config.train_config.gpu >= 0 and model.to_gpu()
models = {'main': model}

optimizer = train_manager.make_optimizer(model, 'main')
optimizers = {'main': optimizer}

if config.model_config.discriminator:
    dis_a = choose_discriminator(config.model_config)
    dis_b = choose_discriminator(config.model_config)
    if config.train_config.gpu >= 0:
        dis_a.to_gpu()
        dis_b.to_gpu()
    models['dis_a'] = dis_a
    models['dis_b'] = dis_b

    optimizer_a = train_manager.make_optimizer(dis_a, 'discriminator')
    optimizer_b = train_manager.make_optimizer(dis_b, 'discriminator')

    optimizers['dis_a'] = optimizer_a
    optimizers['dis_b'] = optimizer_b
else:
    dis_a = dis_b = None

loss_maker = CoEncodeModelLossMaker(config.loss_config, model, dis_a, dis_b)

updater = CoEncodeModelUpdater(
    optimizer=optimizers,
    iterator={'a': iterator_train_a, 'b': iterator_train_b},
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
    converter=lambda batch, device: chainer.dataset.convert.concat_examples(
        [{k: v['input'] for k, v in elem.items()} for elem in batch], device)
)
trainer.run()
