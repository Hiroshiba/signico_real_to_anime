import argparse
import more_itertools
import numpy
import os
import sys

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT_PATH)

from deep_image_converter.config import Config
from deep_image_converter import dataset
from deep_image_converter.drawer import Drawer
from deep_image_converter.model import *
from deep_image_converter.utility.image import save_images, save_tiled_image

parser = argparse.ArgumentParser()
parser.add_argument('path_result_directory')
parser.add_argument('target_iteration', type=int)
parser.add_argument('--num_image', type=int, default=10)
parser.add_argument('--batchsize', type=int, default=10)
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()

path_result_directory = args.path_result_directory

config_path = Config.get_config_path(path_result_directory)
config = Config(config_path)

drawer = Drawer(path_result_directory=path_result_directory, gpu=args.gpu)
assert drawer.load_model(args.target_iteration)

path_images = os.path.join(path_result_directory, 'make_images')
os.path.exists(path_images) or os.mkdir(path_images)

dataset_test = dataset.choose(config.dataset_config)['test']
dataset_train = dataset.choose(config.dataset_config)['train_eval']

# auto_encode
if isinstance(drawer.model, BaseAutoEncoderModel):
    images_ae = []
    for batch in more_itertools.chunked(dataset_test[:args.num_image], args.batchsize):
        images_array = numpy.concatenate([item['input'][numpy.newaxis] for item in batch], axis=0)
        images_ae += drawer.auto_encode(images_array)

    images_ae_train = []
    for batch in more_itertools.chunked(dataset_train[:args.num_image], args.batchsize):
        images_array = numpy.concatenate([item['input'][numpy.newaxis] for item in batch], axis=0)
        images_ae_train += drawer.auto_encode(images_array)

    images_random = []
    for batch in more_itertools.chunked(range(args.num_image), args.batchsize):
        images_random += drawer.random_draw(len(batch))

    path_save = os.path.join(path_images, 'test_{}'.format(args.target_iteration))
    os.path.exists(path_save) or os.mkdir(path_save)

    paths = save_images(images_ae, path_save, 'ae_')
    save_tiled_image(paths, col=args.num_image)

    paths = save_images(images_ae_train, path_save, 'ae_train_')
    save_tiled_image(paths, col=args.num_image)

    paths = save_images(images_random, path_save, 'random_')
    save_tiled_image(paths, col=args.num_image)

# convert
if isinstance(drawer.model, BaseConvertModel):
    images_converted = []
    for batch in more_itertools.chunked(dataset_test[:args.num_image], args.batchsize):
        images_array = numpy.concatenate([item['image_a'][numpy.newaxis] for item in batch], axis=0)
        images_converted += drawer.convert(images_array)

    path_save = os.path.join(path_images, 'test_{}'.format(args.target_iteration))
    os.path.exists(path_save) or os.mkdir(path_save)

    paths = save_images(images_converted, path_save, 'converted_')
    save_tiled_image(paths, col=args.num_image)

# each convert
if isinstance(drawer.model, BaseEachConvertModel):
    images_converted_a = []
    images_converted_b = []
    for batch in more_itertools.chunked(dataset_test[:args.num_image], args.batchsize):
        images_array_a = numpy.concatenate([item['image_a']['input'][numpy.newaxis] for item in batch], axis=0)
        images_array_b = numpy.concatenate([item['image_b']['input'][numpy.newaxis] for item in batch], axis=0)

        images_converted_b += drawer.convert_from_to(images_array_a, 'a', 'b')
        images_converted_a += drawer.convert_from_to(images_array_b, 'b', 'a')

    path_save = os.path.join(path_images, 'test_{}'.format(args.target_iteration))
    os.path.exists(path_save) or os.mkdir(path_save)

    prefix = 'from_{}_to_{}_'.format('b', 'a')
    paths = save_images(images_converted_a, path_save, prefix)
    save_tiled_image(paths, col=args.num_image)

    prefix = 'from_{}_to_{}_'.format('a', 'b')
    paths = save_images(images_converted_b, path_save, prefix)
    save_tiled_image(paths, col=args.num_image)
